import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys
import utils


def get_loss_func(name='DefaultLoss'):
    return getattr(sys.modules[__name__], name)


class DefaultLoss(nn.Module):
    def __init__(self, neg_sample=100, dist_threshold=1.0):
        super().__init__()
        self.neg_sample = neg_sample
        self.dist_threshold = dist_threshold

    def forward(self, in_dict, out_list):
        for i in range(2):
            info = in_dict['pdbs'][i]['surface_raw']
            xyz = info['xyz']
            iface = info['iface'][:, 0]
            pos_index = (iface == 1)
            out_list[i]['pos_xyz'] = xyz[pos_index]
            out_list[i]['pos_feats_t'] = out_list[i]['feats_target'][pos_index]
            out_list[i]['pos_feats_b'] = out_list[i]['feats_binder'][pos_index]
        
        total_pos_preds = []
        total_neg_preds = []
        for i in range(2):
            j = 1 - i
            
            # positive pairs
            dist_mask = utils.cdist_tensor(out_list[i]['pos_xyz'], out_list[j]['pos_xyz']) < self.dist_threshold
            pos_feat_dists = torch.matmul(out_list[i]['pos_feats_t'], out_list[j]['pos_feats_b'].T)

            pos_preds = pos_feat_dists[dist_mask]
            total_pos_preds.append(pos_preds)

            # sample negtive pairs
            sample_choice = np.random.choice(len(out_list[j]['feats_binder']), self.neg_sample, replace=False)
            neb_preds = torch.matmul(out_list[i]['pos_feats_t'], out_list[j]['feats_binder'][sample_choice].T).view(-1)
            total_neg_preds.append(neb_preds)

        pos_preds = torch.cat(total_pos_preds, dim=0)
        pos_labels = torch.ones_like(pos_preds)
        
        neg_preds = torch.cat(total_neg_preds, dim=0)
        sample_choice = np.random.choice(len(neg_preds), len(pos_preds), replace=False)
        neg_preds = neg_preds[sample_choice]
        neg_labels = torch.zeros_like(neg_preds)

        preds = torch.cat([pos_preds, neg_preds], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        loss = F.binary_cross_entropy_with_logits(preds, labels)
        return loss
