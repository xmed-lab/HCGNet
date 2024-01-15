import os
import numpy as np
import copy
import torch
from torch.utils.data import Dataset

from utils import config


class PDB_Atom(Dataset):
    def __init__(self, pdb_ids, load_keys=['atom', 'mesh', 'residue', 'surface'], transform=None):
        super().__init__()
        print(' --- PDB_Atom: len_ids {}, load_keys {}'.format(len(pdb_ids), load_keys))

        self.data_list = []
        verbose = True
        for pid in pdb_ids:
            item = {'id': pid}
            for key in load_keys:
                npz_file = np.load(
                    os.path.join(config.DATA_SITE_SINGLE, pid, '{}.npz'.format(key))
                )
                item[key] = {}
                for tag in npz_file.keys():
                    item[key][tag] = npz_file[tag]
                if verbose:
                    print(key, item[key].keys())
            verbose = False
            self.data_list.append(item)

        self.transform = transform
        print(self.transform)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = copy.deepcopy(self.data_list[index])

        for key in item.keys():
            if key == 'id': continue
            for tag in item[key].keys():
                item[key][tag] = torch.from_numpy(item[key][tag]).float()

        if self.transform:
            item = self.transform(item)

        return item

    @staticmethod
    def collate_fn(batch):
        return batch
