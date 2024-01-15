import os
import torch
import numpy as np
from tqdm import tqdm

import utils
from utils import config, geo_lib, data_io


def load_split(split):
    LIST_DIR = './tasks/pdb_site/lists/'
    pdb_ids = np.loadtxt(
        os.path.join(LIST_DIR, f'{split}.txt'), 
        dtype=str
    ).tolist()
    return pdb_ids


def process_single(all_ids, save_root, skip=False): # will be used in 'search' task preprocessing
    for pdb_id in tqdm(all_ids):
        if skip and os.path.exists(os.path.join(save_root, pdb_id)):
            continue
        
        if not utils.check_pdb_exist(pdb_id):
            print(pdb_id)
            import pdb; pdb.set_trace()
        
        atom_dict = data_io.load_atom(pdb_id)
        mesh_dict = data_io.load_mesh(pdb_id)

        atom_xyz = torch.from_numpy(atom_dict['atom']['xyz']).float().cuda()
        atom_types = torch.from_numpy(atom_dict['atom']['types']).float().cuda()
        atom_batch = torch.full([atom_xyz.shape[0]], 0).long().cuda()
        
        mesh_xyz = torch.from_numpy(mesh_dict['xyz']).float().cuda()
        mesh_iface = torch.from_numpy(mesh_dict['iface']).float().cuda()
        mesh_batch = torch.full([mesh_xyz.shape[0]], 0).long().cuda()

        surf_xyz, surf_normals, surf_batch = geo_lib.atoms_to_points_normals(
            atom_xyz,
            atom_batch,
            atomtypes=atom_types
        )
        surf_labels = geo_lib.project_iface_labels(
            surf_xyz, 
            surf_batch,
            mesh_xyz=mesh_xyz,
            mesh_labels=mesh_iface,
            mesh_batch=mesh_batch
        )
        surf_curv = geo_lib.curvatures(
            vertices=surf_xyz,
            triangles=None,
            scales=[1, 2, 3, 5, 10],
            batch=surf_batch,
            normals=surf_normals
        )
        surf_dict = {
            'xyz': surf_xyz.data.cpu().numpy(),
            'iface': surf_labels.cpu().numpy(),
            'curvatures': surf_curv.cpu().numpy(),
            'normals': surf_normals.cpu().numpy()
        }

        save_dict = {
            'mesh': mesh_dict,               # xyz, iface
            'atom': atom_dict['atom'],       # xyz, types, kd_scale
            'residue': atom_dict['residue'], # xyz, kd_scale
            'surface': surf_dict             # xyz, iface, curvatures, normals
        }

        save_dir = os.path.join(save_root, pdb_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for key, val in save_dict.items():
            save_path = os.path.join(save_dir, '{}.npz'.format(key))
            np.savez(save_path, **val)


if __name__ == '__main__':
    train_ids = load_split('train')
    test_ids = load_split('test')
    all_ids = train_ids + test_ids
    all_ids = sorted(list(set(all_ids)))
    
    process_single(all_ids, save_root=config.DATA_SITE_SINGLE, skip=False)
