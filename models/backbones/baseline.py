import torch
import torch.nn as nn
from models.utils import Atom_Linear, Atom_Query
from models.utils.pointnet2 import SAModule_res, MultiInputSequential


class Backbone(nn.Module):
    def __init__(self, params):
        super().__init__()
        atom_dims = 6
        surf_dims = 10
        self.params = params
        print(params)
        print('---------------------------------\n')
        
        '''
        ========= Atom Net Initialization =========
        '''
        ## linear layer
        atom_in = atom_dims
        print('atom linear:', [atom_in] + params.atom.linear)
        self.atom_linear = Atom_Linear([atom_in] + params.atom.linear)
        
        ## atom channels of every layer
        atom_chs = [
            params.atom.linear[-1]
        ]
        print('  -- atom channels:', atom_chs)
        print('---------------------------------\n')
        
        '''
        ========= Surface Head/SA/FP Module Initialization =========
        '''
        ## head layer
        surf_in = surf_dims + atom_chs[params.surf.head.atom_id]
        head_mlp = [surf_in] + params.surf.head.mlp
        print('head', head_mlp, ' -- r:', params.surf.radius)

        mod_list = []
        for i in range(len(head_mlp) - 1):
            print('  -- res:', i, [head_mlp[i], head_mlp[i+1], head_mlp[i+1]])
            mod_list.append(SAModule_res(
                radius=params.surf.radius,
                nsample=params.surf.nsample,
                channels=[head_mlp[i], head_mlp[i+1], head_mlp[i+1]]
            ))
        self.surf_head = MultiInputSequential(*mod_list)
        self.surf_head_embd = Atom_Query(atom_chs[params.surf.head.atom_id])

        self.out_dim = params.surf.head.mlp[-1]

    def forward(self, input_dict):
        atom_xyz   = input_dict['atom']['xyz'].unsqueeze(0).contiguous()                           # 1, N,  3
        atom_types = input_dict['atom']['types'].unsqueeze(0).transpose(1, 2).contiguous()         # 1, 6,  N (types only)
        surf_xyz   = input_dict['surface']['xyz'].unsqueeze(0).contiguous()                        # 1, M,  3
        surf_curvs = input_dict['surface']['curvatures'].unsqueeze(0).transpose(1, 2).contiguous() # 1, 10, M

        '''
        ========== Atom Feature Embedding ==========
        '''
        atom_feats_list = []
        atom_xyz_list = [atom_xyz]

        # linear layer
        atom_feats = self.atom_linear(atom_types)
        atom_feats_list.append(atom_feats)

        '''
        ========== Surface Feature Learning ==========
        '''
        surf_xyz_list = [surf_xyz]
        surf_feats_list = []

        ## head layer
        atom_feats = atom_feats_list[self.params.surf.head.atom_id]
        atom_xyz = atom_xyz_list[self.params.surf.head.atom_id]
        atom_feats = self.surf_head_embd(atom_feats, atom_xyz, surf_xyz_list[0])
        
        surf_feats = torch.cat([surf_curvs, atom_feats], dim=1)
        surf_feats, _ = self.surf_head(surf_feats, surf_xyz_list[0])
        surf_feats_list.append(surf_feats)

        return surf_feats
