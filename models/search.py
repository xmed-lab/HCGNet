import importlib
import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, params):
        super().__init__()
        BACKBONE = importlib.import_module('models.backbones.' + params.backbone)
        self.backbone = BACKBONE.Backbone(params)
        
        fcn_mlp = [self.backbone.out_dim] + params.fcn.mlp
        print('fcn:', fcn_mlp)

        self.surf_fcn = nn.ModuleList()
        for _ in range(2):
            mod_list = []
            for i in range(len(fcn_mlp) - 1):
                mod_list.append(nn.Sequential(
                    nn.Conv1d(fcn_mlp[i], fcn_mlp[i+1], 1, bias=True),
                    nn.BatchNorm1d(fcn_mlp[i+1]),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                ))
            mod_list.append(nn.Conv1d(fcn_mlp[-1], fcn_mlp[-1], 1, bias=False))
            self.surf_fcn.append(nn.Sequential(*mod_list))

    def forward_one(self, input_dict):
        surf_feats = self.backbone(input_dict)
        surf_out_binder = self.surf_fcn[0](surf_feats) # B, C, N
        surf_out_target = self.surf_fcn[1](surf_feats) # B, C, N
        return {
            'feats_binder': surf_out_binder.squeeze(0).transpose(0, 1),
            'feats_target': surf_out_target.squeeze(0).transpose(0, 1)
        }

    def forward(self, input_dict, forward_single=False):
        ret_list = []
        for i, pdb_dict in enumerate(input_dict['pdbs']):
            if i == 1 and forward_single and self.training:
                self.eval()
                with torch.no_grad():
                    out_dict = self.forward_one(pdb_dict)
                self.train()
            else:
                out_dict = self.forward_one(pdb_dict)
            ret_list.append(out_dict)

        return ret_list
