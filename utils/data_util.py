# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import torch
from torch_geometric.data import Data


def np_one_hot(in_arr, num_classes):
    out_arr = np.zeros((in_arr.size, num_classes), dtype=np.float64)
    out_arr[np.arange(in_arr.size), in_arr] = 1
    return out_arr

def torch_one_hot(in_arr, num_classes):
    in_arr = in_arr.to(torch.int64)
    out_arr = torch.nn.functional.one_hot(in_arr, num_classes=num_classes).float()
    return out_arr

def torch_k_hot(in_arr, num_classes):
    out_arr = torch.zeros(num_classes).float()
    out_arr[in_arr] = 1.0
    return out_arr

class BipartiteData(Data):
    def __init__(self, size_s=None, size_t=None, edge_index=None, edge_attr=None):
        super().__init__()
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.size_s = size_s
        self.size_t = size_t

    def __inc__(self, key, value, *args, **kwargs):
        if 'edge_index' in key:
            return torch.tensor([[self.size_s], [self.size_t]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

def category_to_tracking_name_base(split, category_name):
    """
    Default label mapping from nuScenes to nuScenes tracking classes.
    :param category_name: Generic nuScenes class.
    :return: nuScenes tracking class.
    """
    if split == 'diverse':
        tracking_mapping = {
            'vehicle.bicycle': 'bicycle',
            'vehicle.car': 'car',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.trailer': 'trailer'
        }
    
    elif split == 'urban':
        tracking_mapping = {
            'vehicle.car': 'car',
            'vehicle.motorcycle': 'motorcycle',
            'vehicle.truck': 'truck',
            'vehicle.trailer': 'trailer'
        }

    elif split == 'rare':
        tracking_mapping = {
            'vehicle.car': 'car',
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'vehicle.truck': 'truck',
            'vehicle.trailer': 'trailer'
        }


    if category_name in tracking_mapping:
        return tracking_mapping[category_name]
    else:
        return None


NuScenesClasses = {
    'car' : 0,
    'pedestrian' : 1,
    'bicycle' : 2,
    'truck' : 3,
    'bus' : 4,
    'motorcycle' : 5,
    'trailer' : 6,
}

NuScenesClassesBase = {
    'urban':{
        'car' : 0,
        'truck' : 3,
        'motorcycle' : 5,
        'trailer' : 6,
    },
    'diverse':{
        'car' : 0,
        'bicycle' : 2,
        'bus' : 4,
        'trailer' : 6,
    },
    'rare':{
        'car' : 0,
        'pedestrian' : 1,
        'truck' : 3,
        'trailer' : 6,
    }
}

