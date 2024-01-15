import sys
import torch
import math
from torch import nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from knn_cuda import KNN
import pointnet2.pointnet2_utils as pn2_utils


def get_atom_interp(name='Atom_Query'):
    return getattr(sys.modules[__name__], name)


class Atom_Linear(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        mlp_list = []
        for i in range(len(channels) - 1):
            mlp_list += [
                nn.Conv1d(channels[i], channels[i + 1], 1, bias=True),
                nn.BatchNorm1d(channels[i + 1]),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]
        self.conv = nn.Sequential(*mlp_list)
    
    def forward(self, atom_feats):
        '''
        inputs:
            - atom_feats: B, C, N
        returns:
            - new_feats: B, C', N
        '''
        new_feats = self.conv(atom_feats)
        return new_feats


class Atom_Query(nn.Module):
    def __init__(self, in_ch, k=16):
        super().__init__()

        self.neb_k = k
        self.knn_func = KNN(k=k, transpose_mode=True)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch + 1, in_ch, 1, bias=True),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=True),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_ch * 2, in_ch, 1, bias=True),
            nn.BatchNorm1d(in_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, atom_feats, atom_xyz, surf_xyz):
        '''
        inputs:
            - atom_feats: B, C, N
            - atom_xyz:   B, N, 3
            - surf_xyz:   B, M, 3
        returns:
            - surf_feats: B, C, M
        '''
        # knn neighbor querying
        if atom_xyz.shape[1] < self.neb_k:
            knn_tmp = KNN(k=atom_xyz.shape[1], transpose_mode=True)
            dist, idx = knn_tmp(atom_xyz, surf_xyz)
        else:
            dist, idx = self.knn_func(atom_xyz, surf_xyz)
        dist = dist.unsqueeze(1) # B, 1, M, k
        neb_feats = pn2_utils.grouping_operation(atom_feats, idx.int()) # B, C, M, k
        neb_feats = torch.cat([neb_feats, dist], dim=1) # B, C+1, M, k
        
        # conv1 => sum
        neb_feats = self.conv1(neb_feats) # B, C, M, k
        surf_feats_1 = neb_feats.sum(dim=-1, keepdim=False) # B, C, M

        # conv2 => sum
        neb_feats = self.conv2(neb_feats)
        surf_feats_2 = neb_feats.sum(dim=-1, keepdim=False)

        # conv3
        surf_feats = torch.cat([surf_feats_1, surf_feats_2], dim=1) # B, 2C, M
        surf_feats = self.conv3(surf_feats)

        return surf_feats
