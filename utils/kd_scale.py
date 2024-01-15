ELE2NUM = {'C': 0, 'H': 1, 'O': 2, 'N': 3, 'S': 4, 'SE': 5}

# https://github.com/LPDI-EPFL/masif/blob/2a370518e0d0d0b0d6f153f2f10f6630ae91f149/source/triangulation/computeHydrophobicity.py
KD_SCALE_DICT = {}
KD_SCALE_DICT["ILE"] = 4.5
KD_SCALE_DICT["VAL"] = 4.2
KD_SCALE_DICT["LEU"] = 3.8
KD_SCALE_DICT["PHE"] = 2.8
KD_SCALE_DICT["CYS"] = 2.5
KD_SCALE_DICT["MET"] = 1.9
KD_SCALE_DICT["ALA"] = 1.8
KD_SCALE_DICT["GLY"] = -0.4
KD_SCALE_DICT["THR"] = -0.7
KD_SCALE_DICT["SER"] = -0.8
KD_SCALE_DICT["TRP"] = -0.9
KD_SCALE_DICT["TYR"] = -1.3
KD_SCALE_DICT["PRO"] = -1.6
KD_SCALE_DICT["HIS"] = -3.2
KD_SCALE_DICT["GLU"] = -3.5
KD_SCALE_DICT["GLN"] = -3.5
KD_SCALE_DICT["ASP"] = -3.5
KD_SCALE_DICT["ASN"] = -3.5
KD_SCALE_DICT["LYS"] = -3.9
KD_SCALE_DICT["ARG"] = -4.5

res_keys = [
    'ILE', 'VAL', 'LEU', 'PHE', 'CYS', 
    'MET', 'ALA', 'GLY', 'THR', 'SER',
    'TRP', 'TYR', 'PRO', 'HIS', 'GLU',
    'GLN', 'ASP', 'ASN', 'LYS', 'ARG'
]
RESIDUE2NUM = {}
RESIDUE_UNKNOWN = 20
for idx, key in enumerate(res_keys):
    RESIDUE2NUM[key] = idx
