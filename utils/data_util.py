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

NuScenesClasses = {
    'car' : 0,
    'pedestrian' : 1,
    'bicycle' : 2,
    'truck' : 3,
    'bus' : 4,
    'motorcycle' : 5,
    'trailer' : 6,
}

#-----------------SPLIT 1-----------------------
NuScenesClassesBase = {
    'car' : 0,
    'pedestrian' : 1,
    'bicycle' : 2,
    'bus' : 4,
}

def category_to_tracking_name_base(category_name):
    """
    Default label mapping from nuScenes to nuScenes tracking classes.
    :param category_name: Generic nuScenes class.
    :return: nuScenes tracking class.
    """
    tracking_mapping = {
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
    }

    if category_name in tracking_mapping:
        return tracking_mapping[category_name]
    else:
        return None


#-----------------SPLIT 2-----------------------
NuScenesClassesBase = {
    'car' : 0,
    'pedestrian' : 1,
    'truck' : 3,
}

def category_to_tracking_name_base(category_name):
    """
    Default label mapping from nuScenes to nuScenes tracking classes.
    :param category_name: Generic nuScenes class.
    :return: nuScenes tracking class.
    """
    tracking_mapping = {
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

#-----------------SPLIT 3-----------------------

NuScenesClassesBase = {
    'car' : 0,
    'truck' : 3,
    'bus' : 4,
    'trailer' : 6,
}

def category_to_tracking_name_base(category_name):
    """
    Default label mapping from nuScenes to nuScenes tracking classes.
    :param category_name: Generic nuScenes class.
    :return: nuScenes tracking class.
    """
    tracking_mapping = {
        'vehicle.car': 'car',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.truck': 'truck',
        'vehicle.trailer': 'trailer'
    }

    if category_name in tracking_mapping:
        return tracking_mapping[category_name]
    else:
        return None


#-----------------SPLIT 4-----------------------

NuScenesClassesBase = {
    'car' : 0,
    'truck' : 3,
    'motorcycle' : 5,
    'trailer' : 6,
}

def category_to_tracking_name_base(category_name):
    """
    Default label mapping from nuScenes to nuScenes tracking classes.
    :param category_name: Generic nuScenes class.
    :return: nuScenes tracking class.
    """
    tracking_mapping = {
        'vehicle.car': 'car',
        'vehicle.motorcycle': 'motorcycle',
        'vehicle.truck': 'truck',
        'vehicle.trailer': 'trailer'
    }

    if category_name in tracking_mapping:
        return tracking_mapping[category_name]
    else:
        return None
