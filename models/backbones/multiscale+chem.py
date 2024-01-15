import torch
import torch.nn as nn
from models.utils import Atom_Linear, get_atom_interp
from models.utils.pointnet2 import SAModule, SAModule_res, FPModule, PointSampling, MultiInputSequential


class Backbone(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        atom_dims = 6
        surf_dims = 10
        self.params = params
        print(params)
        print('---------------------------------\n')

        Interp_Func = get_atom_interp(params.get('atom_interp', 'Atom_Query'))
        print(Interp_Func)
        
        '''
        ========= Atom Net Initialization =========
        '''
        ## linear layer
        atom_in = atom_dims
        print('atom linear:', [atom_in] + params.atom.linear)
        self.atom_linear = Atom_Linear([atom_in] + params.atom.linear)

        ## sa layer
        atom_in = params.atom.linear[-1]
        print('atom sa:', [atom_in] + params.atom.sa, ' -- r:', params.atom.radius)
        self.atom_sa = SAModule(
            radius=params.atom.radius,
            nsample=params.atom.nsample, 
            channels=[atom_in] + params.atom.sa
        )

        ## sa+ds layer
        atom_in = params.atom.sa[-1]
        print('atom sa+ds:', [atom_in] + params.atom.sa, ' -- r:', params.atom.radius * 2)
        self.atom_sa_ds = SAModule(
            radius=params.atom.radius * 2,
            nsample=params.atom.nsample, 
            channels=[atom_in] + params.atom.sa
        )

        ## upsampling layer
        atom_in = params.atom.sa[-1] * 2 + params.atom.linear[-1]
        print('atom fp:', [atom_in] + params.atom.fp)
        self.atom_fp = FPModule([atom_in] + params.atom.fp)

        ## atom channels of every layer
        atom_chs = [
            params.atom.linear[-1],
            params.atom.sa[-1],
            params.atom.sa[-1],
            params.atom.fp[-1]
        ]
        print('  -- atom channels:', atom_chs)
        print('---------------------------------\n')
        
        '''
        ========= Surface Head/SA/FP Module Initialization =========
        '''
        ## pooling layer
        self.point_pool = PointSampling(ratio=params.surf.sampling)

        ## head layer
        surf_in = surf_dims + atom_chs[params.surf.head.atom_id]
        head_mlp = [surf_in] + params.surf.head.mlp
        mod_list = []
        print('head', head_mlp, ' -- r:', params.surf.radius)
        for i in range(len(head_mlp) - 1):
            print('  -- res:', i, [head_mlp[i], head_mlp[i+1], head_mlp[i+1]])
            mod_list.append(SAModule_res(
                radius=params.surf.radius,
                nsample=params.surf.nsample,
                channels=[head_mlp[i], head_mlp[i+1], head_mlp[i+1]]
            ))
        self.surf_head = MultiInputSequential(*mod_list)
        self.surf_head_embd = Interp_Func(atom_chs[params.surf.head.atom_id])

        ## sa layer
        surf_out = head_mlp[-1]
        self.surf_sa = nn.ModuleList()
        self.surf_sa_embd = nn.ModuleList()
        surf_ds_dims = [surf_out]
        for i, sa_item in enumerate(params.surf.sa):
            surf_in = surf_out + atom_chs[sa_item.atom_id]
            sa_mlp = [surf_in] + sa_item.mlp
            sa_radius = params.surf.radius * (2 ** (i + 1))
            print('sa', i, sa_mlp, ' -- r:', sa_radius)
            
            mod_list = []
            for j in range(len(sa_mlp) - 1):
                print('  -- res:', j, [sa_mlp[j], sa_mlp[j+1], sa_mlp[j+1]])
                mod_list.append(SAModule_res(
                    radius=sa_radius,
                    nsample=params.surf.nsample,
                    channels=[sa_mlp[j], sa_mlp[j+1], sa_mlp[j+1]]
                ))
            self.surf_sa.append(MultiInputSequential(*mod_list))
            self.surf_sa_embd.append(Interp_Func(
                in_ch=atom_chs[sa_item.atom_id]
            ))
            surf_out = sa_mlp[-1]
            surf_ds_dims.append(surf_out)

        ## fp layer
        self.surf_fp = nn.ModuleList()
        for i, fp_mlp in enumerate(params.surf.fp):
            surf_in = surf_out + surf_ds_dims[-(i+2)]
            surf_out = fp_mlp[-1]
            self.surf_fp.append(FPModule([surf_in] + fp_mlp))
            print('fp', i, [surf_in] + fp_mlp)

        '''
        ========= Surface Classification Layer Initialization =========
        '''
        self.surf_fcn_embd = Interp_Func(atom_chs[params.fcn.atom_id])
        self.out_dim = surf_out + atom_chs[params.fcn.atom_id]

    def forward(self, input_dict):
        atom_xyz   = input_dict['atom']['xyz'].unsqueeze(0).contiguous()                           # 1, N,  3
        atom_types = input_dict['atom']['types'].unsqueeze(0).transpose(1, 2).contiguous()         # 1, 6,  N (types only)
        res_xyz    = input_dict['residue']['xyz'].unsqueeze(0).contiguous()                        # 1, N', 3
        surf_xyz   = input_dict['surface']['xyz'].unsqueeze(0).contiguous()                        # 1, M,  3
        surf_curvs = input_dict['surface']['curvatures'].unsqueeze(0).transpose(1, 2).contiguous() # 1, 10, M

        '''
        ========== Atom Feature Embedding ==========
        '''
        atom_feats_list = []
        atom_xyz_list = [atom_xyz, atom_xyz, res_xyz, atom_xyz]

        # linear layer
        atom_feats = self.atom_linear(atom_types)
        atom_feats_list.append(atom_feats)

        ## sa layer
        atom_feats, _ = self.atom_sa(atom_feats, atom_xyz)
        atom_feats_list.append(atom_feats)

        ## sa+ds layer
        atom_feats, _ = self.atom_sa_ds(atom_feats, atom_xyz, res_xyz)
        atom_feats_list.append(atom_feats)

        ## fp layer
        atom_feats = self.atom_fp(
            atom_xyz, 
            res_xyz, 
            torch.cat(atom_feats_list[:2], dim=1), 
            atom_feats
        )
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

        ## sa layer
        for i in range(len(self.surf_sa)):
            atom_feats = atom_feats_list[self.params.surf.sa[i].atom_id]
            atom_xyz = atom_xyz_list[self.params.surf.sa[i].atom_id]
            
            # pool, atom query
            surf_feats = surf_feats_list[i]
            surf_feats, surf_xyz = self.point_pool(surf_feats, surf_xyz_list[i])
            atom_feats = self.surf_sa_embd[i](atom_feats, atom_xyz, surf_xyz)
            surf_feats = torch.cat([surf_feats, atom_feats], dim=1)
            
            surf_feats, _ = self.surf_sa[i](surf_feats, surf_xyz)
            surf_xyz_list.append(surf_xyz)
            surf_feats_list.append(surf_feats)

        ## fp layer
        for i in range(len(self.surf_fp)):
            surf_feats = self.surf_fp[i](
                surf_xyz_list[-(i+2)],
                surf_xyz_list[-(i+1)],
                surf_feats_list[-(i+2)],
                surf_feats
            )

        ## fcn layer
        atom_feats = atom_feats_list[self.params.fcn.atom_id]
        atom_xyz = atom_xyz_list[self.params.fcn.atom_id]
        atom_feats = self.surf_fcn_embd(atom_feats, atom_xyz, surf_xyz_list[0])

        surf_feats = torch.cat([surf_feats, atom_feats], dim=1)
        return surf_feats
