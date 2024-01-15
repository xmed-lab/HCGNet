import torch
from scipy.spatial.transform import Rotation
import utils

class CenterCoordsPairs(object):
    def __init__(self, keys=['atom', 'surface', 'mesh', 'residue']):
        super().__init__()
        self.keys = keys

    def __call__(self, item):
        atom_xyz = torch.cat([
            item['pdbs'][0]['atom']['xyz'],
            item['pdbs'][1]['atom']['xyz'],
        ])
        offset = torch.mean(atom_xyz, dim=0, keepdim=True)
        for key in self.keys:
            for i in range(2):
                if key in item['pdbs'][i].keys():
                    item['pdbs'][i][key]['xyz'] -= offset
        return item


class RandomRotationPairs(object):
    def __init__(self, keys=['atom', 'surface', 'mesh', 'residue']):
        super().__init__()
        self.keys = keys
    
    def __call__(self, item):
        R = torch.FloatTensor(Rotation.random().as_matrix()).T
        for key in self.keys:
            for i in range(2):
                if key in item['pdbs'][i].keys(): # ignore 'normals' here
                    item['pdbs'][i][key]['xyz'] = torch.matmul(item['pdbs'][i][key]['xyz'], R)
                    if 'normals' in item['pdbs'][i].keys():
                        item['pdbs'][i][key]['normals'] = torch.matmul(item['pdbs'][i][key]['normals'], R)
        return item


class PreComputePositivePairs(object):
    def __init__(self, dist_threshold=1.0):
        super().__init__()
        self.dist_threshold = dist_threshold

    def __call__(self, item):
        pos_xyz_1 = item['pdbs'][0]['surface']['xyz'][item['pdbs'][0]['surface']['iface'][:, 0] == 1]
        pos_xyz_2 = item['pdbs'][1]['surface']['xyz'][item['pdbs'][1]['surface']['iface'][:, 0] == 1]
        pos_xyz_dists = utils.cdist_tensor(pos_xyz_1, pos_xyz_2)
        item['pre_dist_mask'] = pos_xyz_dists < self.dist_threshold
        return item
    