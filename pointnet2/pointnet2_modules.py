import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from pointnet2 import pointnet2_utils as pn2_utils
from pointnet2 import pytorch_utils as torch_utils
from typing import List, Optional, Tuple

class PointnetSAModule(nn.Module):
    def __init__(self, 
                 mlp: List[int], 
                 npoint: Optional[int], 
                 radius: float, 
                 nsample: int,
                 bn: bool = True, 
                 use_xyz: bool = True, 
                 use_leaky: bool = True, 
                 alpha: float = 0.2, 
                 pool_method: str = 'max', 
                 instance_norm: bool = False):
        
        super().__init__()
        self._grouper = pn2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
        if use_xyz:
            mlp[0] += 3
        
        if use_leaky:
            self._mlp = torch_utils.SharedMLP(
                            mlp, 
                            bn=bn, 
                            instance_norm=instance_norm, 
                            activation=nn.LeakyReLU(negative_slope=alpha, inplace=True))
        else:
            self._mlp = torch_utils.SharedMLP(
                            mlp, 
                            bn=bn, 
                            instance_norm=instance_norm)

        self._npoint = npoint
        self._pool_method = pool_method


    def forward(self, 
                xyz: Tensor, 
                features: Tensor = None, 
                npoint: Optional[int] = None, 
                new_xyz: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        
        if npoint is None:
            npoint = self._npoint

        # farthest point sampling
        assert npoint is not None, 'invalid npoint'
        if new_xyz is None:
            new_xyz = pn2_utils.gather_operation(
                xyz.transpose(1, 2).contiguous(), # B, 3, N
                pn2_utils.furthest_point_sample(xyz, npoint) # B, npoint
            ).transpose(1, 2).contiguous() # B, npoint, 3

        # grouping and pooling
        new_features = self._grouper(xyz, new_xyz, features)
        new_features = self._mlp(new_features)
        if self._pool_method == 'max':
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        elif self._pool_method == 'avg':
            new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        else:
            raise NotImplementedError

        return new_xyz, new_features.squeeze(-1)


class PointnetFPModule(nn.Module):
    def __init__(self, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        self._mlp = torch_utils.SharedMLP(mlp, bn=bn)

    def forward(self, 
                unknown: Tensor, 
                known: Tensor, 
                unknow_feats: Optional[Tensor], 
                known_feats: Tensor) -> Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propagated to
        :param known_feats: (B, C2, m) tensor of features to be propagated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        # interpolate featrues (upsampling)
        dist, idx = pn2_utils.three_nn(unknown, known)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interp_feats = pn2_utils.three_interpolate(known_feats, idx, weight)

        # skip connection
        new_features = interp_feats
        if unknow_feats is not None:
            new_features = torch.cat([new_features, unknow_feats], dim=1)  # (B, C2 + C1, n)
        
        # mlp
        new_features = new_features.unsqueeze(-1)
        new_features = self._mlp(new_features)
        return new_features.squeeze(-1)
