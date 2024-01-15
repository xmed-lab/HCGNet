import os
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import pickle

from utils import config
from tasks.pdb_site.processing import process_single


def load_split(split, pairs=True, enable_eval=False, eval_part=0.1):
    LIST_DIR = './tasks/pdb_search/lists/'
    if split in ['eval', 'train']:
        pdb_ids = np.loadtxt(
            os.path.join(LIST_DIR, 'train.txt'), 
            dtype=str
        ).tolist()
    else: # 'test'
        pdb_ids = np.loadtxt(
            os.path.join(LIST_DIR, 'test.txt'),
            dtype=str
        ).tolist()
    
    if not pairs:
        all_ids = []
        for pdb in pdb_ids:
            pdb_splits = pdb.split('_')
            all_ids += [
                pdb_splits[0] + '_' + pdb_splits[1], # first chain
                pdb_splits[0] + '_' + pdb_splits[2]  # second chain
            ]
        pdb_ids = all_ids
    
    if split != 'test' and enable_eval:
        eval_num = int(len(pdb_ids) * eval_part)
        eval_ids = np.arange(0, len(pdb_ids), len(pdb_ids) // eval_num)[:eval_num]

        train_list = []
        eval_list = []
        for i, pid in enumerate(pdb_ids):
            if i in eval_ids:
                eval_list.append(pid)
            else:
                train_list.append(pid)
        
        if split == 'train_eval':
            pdb_ids = (train_list, eval_list)
        elif split == 'eval':
            pdb_ids = eval_list
        elif split == 'train':
            pdb_ids = train_list
    
    return pdb_ids


def load_processed_single(pdb_id, keys):
    ret_dict = {}
    data_dir = os.path.join(config.DATA_SEARCH_SINGLE, pdb_id)
    for key in keys:
        ret_dict[key] = np.load(os.path.join(data_dir, '{}.npz'.format(key)))
    return ret_dict


def process_pair(skip=False, dist_threshold=1.0):
    train_ids = load_split('train', pairs=True)
    test_ids = load_split('test', pairs=True)
    all_ids = sorted(train_ids + test_ids)

    for pair_id in tqdm(all_ids):
        if skip and os.path.exists(os.path.join(config.DATA_SEARCH_PAIR, pair_id)):
            continue
        
        pair_splits = pair_id.split('_')
        pdb_ids = (
            pair_splits[0] + '_' + pair_splits[1],
            pair_splits[0] + '_' + pair_splits[2]
        )

        # load processed proteins
        pdb_datas = []
        for pdb_id in pdb_ids:
            pdb_datas.append(load_processed_single(pdb_id, keys=['surface', 'mesh']))

        # compute mesh and surface complementarity
        save_dict = {}
        for key in ['surface', 'mesh']:
            pdb_dists = cdist(pdb_datas[0][key]['xyz'], pdb_datas[1][key]['xyz'])
            iface_pairs = [
                (pdb_dists.min(axis=1) < dist_threshold).astype(np.int64),
                (pdb_dists.min(axis=0) < dist_threshold).astype(np.int64)
            ]
            pos_mask = pdb_dists < dist_threshold
            save_dict[key] = {
                'pos_mask': np.where(pos_mask),
                'iface_0': iface_pairs[0][:, np.newaxis],
                'iface_1': iface_pairs[1][:, np.newaxis]
            }

        for i in range(2):
            surf_xyz = pdb_datas[i]['surface']['xyz']
            mesh_xyz = pdb_datas[i]['mesh']['xyz']
            mesh_iface = save_dict['mesh']['iface_' + str(i)][:, 0]

            mesh_tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(mesh_xyz)
            dist, index = mesh_tree.kneighbors(surf_xyz)

            surf_iface = mesh_iface[index[:, 0]]
            dist_mask = dist[:, 0] < 2.0 # <-- meaningless threshold
            surf_iface *= dist_mask
            save_dict['surface']['iface_interp_' + str(i)] = surf_iface[:, np.newaxis]
            
        save_dir = os.path.join(config.DATA_SEARCH_PAIR, pair_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for key, val in save_dict.items():
            save_path = os.path.join(save_dir, '{}.npz'.format(key))
            np.savez(save_path, **val)


def pickle_datas(split='test'):
    pair_ids = load_split(split, pairs=True, enable_eval=True, eval_part=0.1)
    load_keys = {
        'mesh': ['xyz'],
        'atom': ['xyz', 'types', 'kd_scale'],
        'residue': ['xyz'],
        'surface': ['xyz', 'curvatures', 'normals']
    }

    pdb_list = {}
    pair_list = []
    for pair_id in tqdm(pair_ids):
        splits = pair_id.split('_')
        pdb_ids = [
            splits[0] + '_' + splits[1],
            splits[0] + '_' + splits[2]
        ]
        for pdb_id in pdb_ids:
            if pdb_id in pdb_list.keys(): # skip if loaded already
                continue
            
            pdb_list[pdb_id] = {}
            for key in load_keys.keys():
                pdb_list[pdb_id][key] = {}
                npz_file = np.load(os.path.join(config.DATA_SEARCH_SINGLE, pdb_id, '{}.npz'.format(key)))
                for tag in load_keys[key]:
                    pdb_list[pdb_id][key][tag] = npz_file[tag]

        pair_info = {}
        for key in ['surface', 'mesh']:
            pair_info[key] = np.load(os.path.join(config.DATA_SEARCH_PAIR, pair_id, '{}.npz'.format(key)))
        
        if split != 'test' and \
           (pair_info['surface']['iface_0'].shape[0] > 25000 or \
            pair_info['surface']['iface_1'].shape[0] > 25000):
            print('skip:', pair_id)
            continue
        
        pair_list.append({
            'pair_id': pair_id,
            'pdbs': [
                {
                    'id': pdb_ids[0],
                    'surface': {
                        'iface': pair_info['surface']['iface_0'],
                        'iface_interp': pair_info['surface']['iface_interp_0']
                    },
                    'mesh': {
                        'iface': pair_info['mesh']['iface_0']
                    }
                },
                {
                    'id': pdb_ids[1],
                    'surface': {
                        'iface': pair_info['surface']['iface_1'],
                        'iface_interp': pair_info['surface']['iface_interp_1']
                    },
                    'mesh': {
                        'iface': pair_info['mesh']['iface_1']
                    }
                }
            ]
        })

    with open(os.path.join(config.DATA_SEARCH_PICKLE, '{}.pickle'.format(split)), 'wb') as f:
        pickle.dump(pdb_list, f, protocol=0)
        pickle.dump(pair_list, f, protocol=0)
        

if __name__ == '__main__':
    train_ids = load_split('train', pairs=False)
    test_ids = load_split('test', pairs=False)
    all_ids = train_ids + test_ids
    all_ids = sorted(list(set(all_ids)))

    print(' ---- process single')
    process_single(all_ids, save_root=config.DATA_SEARCH_SINGLE, skip=False)
    
    print(' ---- process pair')
    process_pair(skip=False)

    print(' ---- process pickle')
    pickle_datas(split='train')
    pickle_datas(split='eval')
    pickle_datas(split='test')
