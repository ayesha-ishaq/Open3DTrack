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

def category_to_tracking_name_base(category_name):
    """
    Default label mapping from nuScenes to nuScenes tracking classes.
    :param category_name: Generic nuScenes class.
    :return: nuScenes tracking class.
    """
    tracking_mapping = {
        'vehicle.bicycle': 'bicycle',
        'vehicle.car': 'car',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'vehicle.truck': 'truck'
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
    'car' : 0,
    'bicycle' : 2,
    'truck' : 3,
    'trailer' : 6,
}

# NOVEL_LABELS = ['pedestrian','bus','motorcycle']
# MAPPER = [1, 4, 5]

NOVEL_LABELS = ['pedestrian', 'person','bus', 'motorcycle', 'motorbike', 'scooter', 'bike']
MAPPER = [1, 1, 4, 5, 5, 5, 5]

# NUSCENES_LABELS = ['car', 'person', 'pedestrian', 'bicycle', 'truck', 'bus', 'motorcycle', 'motorbike', 'scooter', 'trailer', 'semi trailer', 'cargo container', 'shipping container', 'freight container', 'camper', 'recreational vehicle']
# MAPPER = [0, 1, 1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6]