# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from torch import Tensor
from copy import deepcopy
import iou3d_nms_cuda

from torch_geometric.data import Batch
from torch_geometric.utils import unbatch

from utils.data_util import BipartiteData, NuScenesClassesBase


def adj_to_edge_index(adj):
    return torch.nonzero(adj).transpose(1, 0).long()

def _velocity_adj(boxes, time_diff=0.5):

    center_dist = torch.cdist(boxes[0][:, :2], boxes[1][:, :2], p=2.0)
    distance_thresh = 3

    adj = torch.le(center_dist, distance_thresh).int()
    return adj


def velocity_adj(boxes, velo, time_diff=0.5):

    track_boxes = boxes[0]
    detection_boxes = boxes[1]

    center_dist = torch.cdist(track_boxes[:, :2], detection_boxes[:, :2], p=2.0)
    velo_thresh = torch.norm(velo[0].detach() - velo[1], p=2, dim=1)*(0.1) + 3
    adj = torch.le(center_dist, velo_thresh.unsqueeze(1)).int()
    
    return adj

def bev_euclidean_distance_adj(boxes, classes=None, thresh=2.0):
    if isinstance(boxes, Tensor):
        boxes = (boxes, boxes)
    
    center_dist = torch.cdist(boxes[0][:, :2],
                              boxes[1][:, :2], p=2.0)
    adj = torch.le(center_dist, thresh).int()

    if classes is not None:
        if isinstance(classes, Tensor):
            classes = (classes, classes)
        cls_mask = torch.eq(classes[0].unsqueeze(1),
                            classes[1].unsqueeze(0))
        unknown_1 = classes[0].unsqueeze(1) == 7
        unknown_2 = classes[1].unsqueeze(0) == 7
        
        combined_unknown_mask = unknown_1 | unknown_2
        adj *= torch.logical_or(cls_mask, combined_unknown_mask).int()

    return adj

def class_identical_adj(classes):
    if isinstance(classes, Tensor):
        classes = (classes, classes)
    adj = torch.eq(classes[0].unsqueeze(1), classes[1].unsqueeze(0))
    return adj.int()

def fully_connected_adj(size_a, size_b):
    return torch.ones(size_a, size_b, dtype=torch.int32)


def build_inter_graph(det_boxes, track_boxes, track_velo, track_calc_velo,
                      track_age, det_batch, track_batch
                      ):

    det_boxes_list = unbatch(det_boxes, det_batch)
    
    track_boxes_list = unbatch(track_boxes, track_batch)
    track_velo_list = unbatch(track_velo, track_batch)
    track_calc_velo_list = unbatch(track_calc_velo, track_batch)

    track_age_list = unbatch(track_age, track_batch)

    data_list = []
    for boxes_d, boxes_t, age_t, velo_t, velo_c_t in zip(det_boxes_list,
        track_boxes_list, track_age_list, track_velo_list,
        track_calc_velo_list
        ):

        # Move track boxes forward using their velocity
        pred_boxes_t = deepcopy(boxes_t)
        # TODO: update with precise time stamp
        pred_boxes_t[:, :2] += velo_t.detach() * age_t.unsqueeze(1) * 0.5 # s1 = s0 + v0 * delta_t

        # adj = velocity_adj((pred_boxes_t, boxes_d), (velo_t, velo_c_t))
        adj = _velocity_adj((pred_boxes_t, boxes_d))
        edge_index = adj_to_edge_index(adj)

        diff_box = boxes_d[edge_index[1, :], :] - boxes_t[edge_index[0, :], :]
        diff_time = age_t[edge_index[0, :]].unsqueeze(1)
        diff_position_pred = boxes_d[edge_index[1, :], :2] - pred_boxes_t[edge_index[0, :], :2]

        # Initial edge attribute
        # frame time difference, position difference, size difference
        # and the differences in the predicted position assuming constant velocity
        edge_attr = torch.cat([diff_box, diff_position_pred, diff_time], dim=1)

        size_s = boxes_t.size(0)
        size_t = boxes_d.size(0)
        
        data = BipartiteData(size_s=size_s, size_t=size_t,
                             edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(data)
    
    data_batch = Batch.from_data_list(data_list, follow_batch=['edge_index'])

    return data_batch
