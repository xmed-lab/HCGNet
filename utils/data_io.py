import os
import numpy as np
from Bio import PDB
from plyfile import PlyData

from utils import config
from utils.kd_scale import ELE2NUM, KD_SCALE_DICT, RESIDUE2NUM, RESIDUE_UNKNOWN


def load_atom(pdb_id):
    pdb_path = os.path.join(config.DATA_RAW, '01-benchmark_pdbs', '{}.pdb'.format(pdb_id))
    parser = PDB.PDBParser()
    protein = parser.get_structure('structure', pdb_path)

    atom_xyz = []
    atom_types = []
    atom_kd_scale = []
    res_xyz = []
    res_types = []
    res_kd_scale = []
    for res in protein.get_residues():
        res_coords = np.mean([atom.get_coord() for atom in res.get_atoms()], axis=0)
        kd_scale = KD_SCALE_DICT.get(res.get_resname(), 0.0) / 4.5
        res_type = np.zeros(RESIDUE_UNKNOWN + 1)
        res_type[RESIDUE2NUM.get(res.get_resname(), RESIDUE2NUM)] = 1

        res_xyz.append(res_coords)
        res_types.append(res_type)
        res_kd_scale.append(kd_scale)

        for atom in res.get_atoms():
            tmp_type = np.zeros(len(ELE2NUM))
            tmp_type[ELE2NUM[atom.element]] = 1.0
            atom_xyz.append(atom.get_coord())
            atom_types.append(tmp_type)
            atom_kd_scale.append(kd_scale)

    return {
        'residue': {
            'xyz': np.array(res_xyz),                         # M, 3
            'types': np.array(res_types),                     # M, 21
            'kd_scale': np.array(res_kd_scale)[:, np.newaxis] # M, 1
        },
        'atom': {
            'xyz': np.array(atom_xyz),                         # N, 3
            'types': np.array(atom_types),                     # N, 6 (one-hot)
            'kd_scale': np.array(atom_kd_scale)[:, np.newaxis] # N, 1
        }
    }


def load_mesh(pdb_id):
    ply_path = os.path.join(config.DATA_RAW, '01-benchmark_surfaces', '{}.ply'.format(pdb_id))
    plydata = PlyData.read(ply_path)
    mesh_xyz = np.vstack([[v[0], v[1], v[2]] for v in plydata['vertex']])
    mesh_iface = plydata['vertex']['iface']
    return {
        'xyz': mesh_xyz,                   # N, 3
        'iface': mesh_iface[:, np.newaxis] # N, 1
    }
