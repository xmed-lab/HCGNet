import os
import torch
from torch import nn
import numpy as np
from utils import config


def check_pdb_exist(pdb_id):
    pdb_path = os.path.join(config.DATA_RAW, '01-benchmark_pdbs', '{}.pdb'.format(pdb_id))
    return os.path.exists(pdb_path)


def iface_valid_filter(item, item_inv):
    # https://github.com/FreyrS/dMaSIF/blob/master/data.py#L22
    
    labels = item['mesh']['labels']
    labels_inv = item_inv['mesh']['labels']
    valid = (
        (torch.sum(labels) < 0.75 * len(labels))
        and (torch.sum(labels) > 30)
        and (torch.sum(labels) > 0.01 * labels_inv.shape[0])
    )
    valid_inv = (
        (torch.sum(labels_inv) < 0.75 * len(labels_inv))
        and (torch.sum(labels_inv) > 30)
        and (torch.sum(labels_inv) > 0.01 * labels.shape[0])
    )
    return valid and valid_inv


def convert_instance_norm(module):
    module_output = module
    if isinstance(module, nn.BatchNorm1d):
        module_output = nn.InstanceNorm1d(module.num_features)
    elif isinstance(module, nn.BatchNorm2d):
        module_output = nn.InstanceNorm2d(module.num_features)
    elif isinstance(module, nn.BatchNorm3d):
        module_output = nn.InstanceNorm3d(module.num_features)
    for name, child in module.named_children():
        module_output.add_module(
            name, convert_instance_norm(child)
        )
    del module
    return module_output


def cdist_tensor(a, b):
    # a: N, 3
    # b: M, 3
    # returns: (N, M)
    return torch.sum((a[:, None, :] - b[None, :, :]) ** 2, dim=-1).sqrt()


def dict_to_cuda(in_dict):
    for key in in_dict.keys():
        if key == 'id': continue
        for tag in in_dict[key].keys():
            in_dict[key][tag] = in_dict[key][tag].cuda()
    return in_dict


def dict_to_cuda_pair(in_dict):
    for i in range(len(in_dict['pdbs'])):
        in_dict['pdbs'][i] = dict_to_cuda(in_dict['pdbs'][i])
    return in_dict


def get_optimizer(model, name, lr, max_epoch):
    lr_scheduler = None
    if name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            amsgrad=True
        )
    elif name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr,
            weight_decay=1e-3
        )
    elif name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=lr,
            momentum=0.98, 
            weight_decay=1e-3,
            nesterov=True
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=1, 
            gamma=np.power(0.001, 1 / max_epoch)
        )
    else:
        raise NotImplementedError

    return optimizer, lr_scheduler


def save_model(model, epoch, save_dir):
    save_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict()
        },
        os.path.join(save_dir, 'epoch_{}.pth'.format(epoch))
    )