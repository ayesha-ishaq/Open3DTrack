# ------------------------------------------------------------------------
# Pytorch Template Project (https://github.com/victoresque/pytorch-template)
# Copyright (c) 2018 Victor Huang. All Rights Reserved.
# ------------------------------------------------------------------------

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def load_clip(model_name="ViT-L/14@336px"):

    print("Loading CLIP {} model...".format(model_name))
    clip_pretrained, _ = clip.load(model_name, device='cuda', jit=False)
    print("Finish loading")
    
    return clip_pretrained

def extract_clip_feature(labelset, clip_pretrained):
    # "ViT-L/14@336px" # the big model that OpenSeg uses
    # print("Loading CLIP {} model...".format(model_name))
    # clip_pretrained, _ = clip.load(model_name, device='cuda', jit=False)
    # print("Finish loading")

    if isinstance(labelset, str):
        lines = labelset.split(',')
    elif isinstance(labelset, list):
        lines = labelset
    else:
        raise NotImplementedError

    labels = []
    for line in lines:
        label = line
        labels.append(label)
    text = clip.tokenize(labels)
    text = text.cuda()
    text_features = clip_pretrained.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features

def farthest_point_sampling(points, num_samples):
    """
    Perform farthest point sampling on a set of points.
    
    Parameters:
    points (np.ndarray): The array of points with shape (N, D) where N is the number of points and D is the dimension.
    num_samples (int): The number of points to sample.
    
    Returns:
    np.ndarray: The indices of the sampled points.
    """
    N, D = points.shape
    sampled_indices = np.zeros(num_samples, dtype=int)
    
    # Initialize distances with infinity
    distances = np.full(N, np.inf)
    
    # Randomly select the first point
    sampled_indices[0] = np.random.randint(N)
    farthest_point = points[sampled_indices[0]]
    
    for i in range(1, num_samples):
        # Update distances with the newly selected point
        dist = np.sum((points - farthest_point) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        
        # Select the farthest point
        sampled_indices[i] = np.argmax(distances)
        farthest_point = points[sampled_indices[i]]
    
    return sampled_indices

def zero_pad_point_cloud(points, fixed_size):
    """
    Zero-pad a set of point cloud points to a fixed size.
    
    Parameters:
    points (np.ndarray): The array of points with shape (N, D) where N is the number of points and D is the dimension.
    fixed_size (int): The fixed size to pad the points to.
    
    Returns:
    np.ndarray: The zero-padded array of points with shape (fixed_size, D).
    """
    N, D = points.shape
    
    if N > fixed_size:
        raise ValueError(f"The number of points ({N}) exceeds the fixed size ({fixed_size}).")
    
    # Create an array of zeros with the fixed size
    padded_points = np.zeros((fixed_size, D))
    
    # Copy the original points to the padded array
    padded_points[:N, :] = points
    
    return padded_points

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


