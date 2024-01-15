import torch
import torch.nn as nn
import torch.nn.functional as F
import pointnet2.pointnet2_utils as pn2_utils


class SAModule(nn.Module):
    def __init__(self, radius, nsample, channels):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        channels[0] += 3
        
        self.conv = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(channels[i - 1], channels[i], 1),
                nn.BatchNorm2d(channels[i]),
                nn.LeakyReLU(negative_slope=0.2),
            )
            for i in range(1, len(channels))
        ])
        self.pool = nn.MaxPool2d([1, nsample])

    def forward(self, feats, xyz, new_xyz=None):
        '''
        inputs:
            - feats:   B, C, N
            - xyz:     B, N, 3
            - new_xyz: B, M, 3 or None
        returns:
            - out_feats: B, C', M
        '''
        if new_xyz is None:
            new_xyz = xyz
        
        xyz_q = xyz.transpose(1, 2).contiguous() # B, 3, N
        new_xyz_q = new_xyz.transpose(1, 2).contiguous() # B, 3, M

        neb_ids = pn2_utils.ball_query(self.radius, self.nsample, xyz, new_xyz) # 1, M, k
        neb_feats = pn2_utils.grouping_operation(feats, neb_ids) # 1, C, M, k
        neb_xyz = pn2_utils.grouping_operation(xyz_q, neb_ids) # 1, 3, M, k
        neb_xyz = (neb_xyz - new_xyz_q.unsqueeze(-1)) / self.radius # <-- rotation

        neb_feats = torch.cat([
            neb_feats,
            neb_xyz
        ], dim=1) # 1, C+3, M, k
        neb_feats = self.conv(neb_feats) # 1, C, M, k
        
        out_feats = self.pool(neb_feats) # 1, C, M, 1
        out_feats = out_feats.squeeze(-1).squeeze(-1) # 1, C, M
        return out_feats, new_xyz


class SAModule_res(nn.Module):
    def __init__(self, radius, nsample, channels):
        super().__init__()
        if channels[0] != channels[-1]:
            self.linear = nn.Conv1d(channels[0], channels[-1], kernel_size=1, bias=False)
        else:
            self.linear = None
        self.conv = SAModule(radius, nsample, channels)

    def forward(self, feats, xyz):
        '''
        inputs:
            - feats: B, C, N
            - xyz:   B, 3, N
        returns:
            - new_feats: B, C', N
        '''
        new_feats, _ = self.conv(feats, xyz)
        if self.linear:
            feats = self.linear(feats)

        new_feats = F.relu(feats + new_feats)
        return new_feats, xyz


class PointSampling(nn.Module):
    def __init__(self, ratio=0.5):
        super().__init__()
        self.ratio = ratio

    def forward(self, feats, xyz):
        '''
        inputs:
            - feats: B, C, N
            - xyz:   B, N, 3
        returns:
            - new_feats: B, C, M
            - new_xyz:   B, M, 3
        '''
        npoint = int(xyz.shape[1] * self.ratio)
        sample_idx = pn2_utils.furthest_point_sample(xyz, npoint) # B, M
        new_feats = pn2_utils.gather_operation(feats, sample_idx) # B, C, M
        new_xyz = pn2_utils.gather_operation(
            xyz.transpose(1, 2).contiguous(),
            sample_idx
        ).transpose(1, 2).contiguous() # B, M, 3

        return new_feats, new_xyz


class FPModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(channels[i - 1], channels[i], 1),
                nn.BatchNorm2d(channels[i]),
                nn.LeakyReLU(negative_slope=0.2),
            )
            for i in range(1, len(channels))
        ])

    def forward(self, unknown, known, unknow_feats, known_feats):
        """
        inputs:
            - unknown: B, N, 3
            - known:   B, M, 3
            - unknown_feats: B, C, N
            - known_feats:   B, C, M
        returns:
            - new_feats: B, C', N
        """
        # upsampling (interpolation)
        dist, idx = pn2_utils.three_nn(unknown, known)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interp_feats = pn2_utils.three_interpolate(known_feats, idx, weight)

        # skip connection
        new_feats = interp_feats
        if unknow_feats is not None:
            new_feats = torch.cat([new_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        
        # mlp
        new_feats = new_feats.unsqueeze(-1)
        new_feats = self.conv(new_feats)
        return new_feats.squeeze(-1)


class MultiInputSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input
