import importlib
from torch import nn


class Network(nn.Module):
    def __init__(self, params):
        super().__init__()
        BACKBONE = importlib.import_module('models.backbones.' + params.backbone)
        self.backbone = BACKBONE.Backbone(params)
        
        n_class = 1
        fcn_mlp = [self.backbone.out_dim] + params.fcn.mlp
        print('fcn:', fcn_mlp)
        mod_list = []
        for i in range(len(fcn_mlp) - 1):
            mod_list.append(nn.Sequential(
                nn.Conv1d(fcn_mlp[i], fcn_mlp[i+1], 1, bias=True),
                nn.BatchNorm1d(fcn_mlp[i+1]),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ))
        mod_list.append(nn.Conv1d(fcn_mlp[-1], n_class, 1, bias=True))
        self.surf_fcn = nn.Sequential(*mod_list)

    def forward(self, input_dict):
        surf_feats = self.backbone(input_dict)
        surf_out = self.surf_fcn(surf_feats)
        return {
            'out_cls': surf_out.squeeze(0).transpose(0, 1)
        }
