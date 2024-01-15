import os
import torch
import copy
import pickle

from torch.utils.data import Dataset
from utils import config


class PDB_Pair(Dataset):
    def __init__(self, split='train', pdb_swap=False, pdb_trans=None, pair_trans=None, post_trans=None):
        super().__init__()
        with open(os.path.join(config.DATA_SEARCH_PICKLE, f'{split}.pickle'), 'rb') as f:
            self.pdb_list = pickle.load(f)
            self.pair_list = pickle.load(f)

        print('dataset: {}, {}, swap: {}'.format(split, len(self.pair_list), pdb_swap))
        self.pdb_swap = pdb_swap
        self.pair_visit = {}
        self.pdb_trans = pdb_trans
        self.pair_trans = pair_trans
        self.post_trans = post_trans
        print(self.pdb_trans)
        print(self.pair_trans)
        print(self.post_trans)

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, index):
        pair_info = self.pair_list[index]
        pair_id = pair_info['pair_id']
        ret_dict = {
            'pair_id': pair_id,
            'pdbs': []
        }
        
        if not self.pdb_swap:
            visit = False
        else:
            visit = self.pair_visit.get(pair_id, False)
            self.pair_visit[pair_id] = (not visit)
        for i in range(2):
            if visit:
                i = 1 - i
            item = pair_info['pdbs'][i]
            '''
            pair_info['pdbs'][x].keys = {
                'id': str,
                'surface': ['iface', 'iface_interp'],
                'mesh': ['iface']
            }
            pdb_keys = {
                'mesh': ['xyz'],
                'atom': ['xyz', 'types', 'kd_scale'],
                'residue': ['xyz'],
                'surface': ['xyz', 'curvatures']
            } # '''
            pdb_info = copy.deepcopy(self.pdb_list[item['id']])
            pdb_info['surface']['iface'] = copy.deepcopy(item['surface']['iface'])
            pdb_info['mesh']['iface'] = copy.deepcopy(item['mesh']['iface']) # for evaluation
            
            for key in pdb_info.keys(): # convert to torch.tensor for further transformation (augmentation)
                for tag in pdb_info[key].keys():
                    pdb_info[key][tag] = torch.from_numpy(pdb_info[key][tag]).float()

            pdb_info['id'] = item['id']
            if self.pdb_trans:
                pdb_info = self.pdb_trans(pdb_info)
            ret_dict['pdbs'].append(pdb_info)

        if self.pair_trans:
            ret_dict = self.pair_trans(ret_dict)

        for i in range(2):
            ret_dict['pdbs'][i]['surface_raw'] = copy.deepcopy(ret_dict['pdbs'][i]['surface'])
            if self.post_trans:
                ret_dict['pdbs'][i] = self.post_trans(ret_dict['pdbs'][i])

        return ret_dict

    @staticmethod
    def collate_fn(batch):
        return batch
