import torch
from scipy.spatial.transform import Rotation
import numpy as np


class CenterCoords(object):
    def __init__(self, keys=['atom', 'surface', 'mesh', 'residue']):
        super().__init__()
        self.keys = keys

    def __call__(self, item):
        atom_center = item['atom']['xyz'].mean(dim=0, keepdim=True)
        for key in self.keys:
            if key in item.keys():
                item[key]['xyz'] -= atom_center
        return item


class RandomRotation(object):
    def __init__(self, keys=['atom', 'surface', 'mesh', 'residue']):
        super().__init__()
        self.keys = keys
    
    def __call__(self, item):
        R = torch.FloatTensor(Rotation.random().as_matrix()).T
        for key in self.keys:
            if key in item.keys():
                item[key]['xyz'] = torch.matmul(item[key]['xyz'], R,)
                if 'normals' in item[key].keys():
                    item[key]['normals'] = torch.matmul(item[key]['normals'], R)
        
        return item


class RandomPermutation(object):
    def __init__(self, keys=['surface', 'atom']):
        super().__init__()
        self.keys = keys

    def __call__(self, item):
        for key in self.keys:
            if key in item.keys():
                list_size = item[key]['xyz'].shape[0]
                shuffle_ids = np.random.choice(list_size, list_size, replace=False)
                for tag in item[key].keys():
                    item[key][tag] = item[key][tag][shuffle_ids]
        return item


class RandomCropPointCloud(object):
    def __init__(self, ratio=0.8, keys=['surface']):
        super().__init__()
        self.ratio = ratio
        self.keys = keys
        assert ratio <= 1.0 and ratio >= 0.0, 'invalid value.'

    def __call__(self, item):
        for _ in range(10):
            boundary = []
            for i in range(3): # x/y/z
                min_x = torch.min(item['surface']['xyz'][:, i]).item()
                max_x = torch.max(item['surface']['xyz'][:, i]).item()

                crop_ratio = np.random.rand() * (1 - self.ratio) + self.ratio # <-- crop_v2
                left = np.random.rand() * (1 - crop_ratio)
                right = left + crop_ratio

                left_x = left * (max_x - min_x) + min_x - 1e-3
                right_x = right * (max_x - min_x) + min_x + 1e-3
                boundary.append((left_x, right_x))

            # check valid
            valid = True
            masks = {}
            for key in self.keys:
                if key in item.keys():
                    mask = np.ones(item[key]['xyz'].shape[0]).astype(bool)
                    for i in range(3):
                        left_x, right_x = boundary[i]
                        points_x = item[key]['xyz'][:, i].numpy()
                        mask = mask & (points_x >= left_x) & (points_x <= right_x)
                    
                    masks[key] = mask
                    if np.sum(mask) <= 0.05 * len(mask):
                        print('invlid cropping')
                        print(np.sum(mask), boundary, key, item[key]['xyz'].shape[0])
                        valid = False
                        break
            
            if not valid:
                continue
            
            for key in self.keys:
                if key in item.keys():
                    mask = masks[key]
                    for tag in item[key].keys():
                        item[key][tag] = item[key][tag][mask]
            break

        return item


class RandomScale(object):
    def __init__(self, scale_low=0.7, scale_high=1.3, keys=['surface', 'atom', 'mesh', 'residue']):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.keys = keys

    def __call__(self, item):
        scale = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[1, 3])
        scale = torch.from_numpy(scale).float()
        for key in item.keys():
            if key in self.keys:
                item[key]['xyz'] *= scale

        return item


class RandomJitter(object):
    def __init__(self, std=0.01, clip=0.1, keys=['surface', 'atom']):
        self.std = std
        self.clip = clip
        self.keys = keys

    def __call__(self, item):
        for key in item.keys():
            if key in self.keys:
                point_size = item[key]['xyz'].shape[0]
                offsets = np.random.normal(scale=self.std, size=[point_size, 3])
                offsets = np.clip(offsets, a_min=-self.clip, a_max=self.clip)
                offsets = torch.from_numpy(offsets).float()
                item[key]['xyz'] += offsets
        
        return item


class RandomFlip(object):
    def __init__(self, keys=['atom', 'surface', 'mesh', 'residue']):
        self.keys = keys

    def __call__(self, item):
        for i in range(3):
            p = np.random.rand()
            if p >= 0.5:
                for key in item.keys():
                    if key in self.keys:
                        item[key]['xyz'][:, i] = -item[key]['xyz'][:, i]
                        if 'normals' in item[key].keys():
                            item[key]['normals'][:, i] = -item[key]['normals'][:, i]
        return item
