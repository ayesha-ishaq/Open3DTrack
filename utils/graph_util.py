# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from torch import Tensor
from copy import deepcopy
import iou3d_nms_cuda

from torch_geometric.data import Batch
from torch_geometric.utils.unbatch import unbatch

from utils.data_util import BipartiteData, NuScenesClassesBase


def adj_to_edge_index(adj):
    return torch.nonzero(adj).transpose(1, 0).long()

def class_velocity_adj(boxes, classes, time_diff=0.5):

    center_dist = torch.cdist(boxes[0][:, :2], boxes[1][:, :2], p=2.0)
    distance_thresh = 3

    adj = torch.le(center_dist, distance_thresh).int() 
 
    return adj

def embedding_velocity_adj(boxes, embeddings, time_diff=0.5):
    center_dist = torch.cdist(boxes[0][:, :2], boxes[1][:, :2], p=2.0)
    embedding_dist = torch.cdist(embeddings[0], embeddings[1], p=2.0)
    center_dist += embedding_dist

    distance_thresh = 35

    adj = torch.le(center_dist, distance_thresh).int()
    return adj

def velocity_adj(boxes, velo, time_diff=0.5):

    # track_boxes = boxes[0]
    # detection_boxes = boxes[1]

    # center_dist = torch.cdist(track_boxes[:, :2], detection_boxes[:, :2], p=2.0)
    # velo_thresh = (torch.norm(velo[0].detach(), p=2, dim=1))*(0.1) + 3

    # adj = torch.le(center_dist, velo_thresh.unsqueeze(1)).int()

    track_boxes = boxes[0]
    detection_boxes = boxes[1]

    # Compute distances over each component (x and y)
    dist_x = torch.cdist(track_boxes[:, 0:1], detection_boxes[:, 0:1], p=2.0)
    dist_y = torch.cdist(track_boxes[:, 1:2], detection_boxes[:, 1:2], p=2.0)

    # Compute velocity threshold for each component
    velo_thresh_x = (torch.abs(velo[0][:, 0].detach()) * (0.1) + 3)
    velo_thresh_y = (torch.abs(velo[0][:, 1].detach()) * (0.1) + 3)

    # Combine thresholds and distances to create a combined threshold check
    adj_x = torch.le(dist_x, velo_thresh_x.unsqueeze(1)).int()
    adj_y = torch.le(dist_y, velo_thresh_y.unsqueeze(1)).int()

    # Combined adjustment matrix (1 where both x and y are within thresholds)
    adj = adj_x * adj_y
    
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

def bev_euclidean_embedding_distance_adj(boxes, embeddings=None, thresh=2.0):
    if isinstance(boxes, Tensor):
        boxes = (boxes, boxes)
    
    center_dist = torch.cdist(boxes[0][:, :2],
                              boxes[1][:, :2], p=2.0)
    adj = torch.le(center_dist, thresh).int()


    if embeddings is not None:
        if isinstance(embeddings, Tensor):
            embeddings = (embeddings, embeddings)
        center_dist = torch.cdist(embeddings[0],
                              embeddings[1], p=2.0)
        embedding_thresh = 30
        cls_mask = torch.le(center_dist, embedding_thresh).int()

        adj *= cls_mask.int()

    return adj

def class_identical_adj(classes):
    if isinstance(classes, Tensor):
        classes = (classes, classes)
    adj = torch.eq(classes[0].unsqueeze(1), classes[1].unsqueeze(0))
    return adj.int()

def fully_connected_adj(size_a, size_b):
    return torch.ones(size_a, size_b, dtype=torch.int32)


def build_inter_graph(det_boxes, track_boxes, det_class, track_class,
                      track_velo, det_velo, track_age, det_batch, track_batch,
                      ):

    det_boxes_list = unbatch(det_boxes, det_batch)
    det_class_list = unbatch(det_class, det_batch)
    det_velo_list = unbatch(det_velo, det_batch)

    track_boxes_list = unbatch(track_boxes, track_batch)
    track_class_list = unbatch(track_class, track_batch)
    track_velo_list = unbatch(track_velo, track_batch)

    track_age_list = unbatch(track_age, track_batch)

    data_list = []
    for boxes_d, cls_d, boxes_t, cls_t, age_t, velo_t, velo_d in zip(det_boxes_list,
        det_class_list, track_boxes_list, track_class_list, track_age_list, track_velo_list,
        det_velo_list
        ):

        # Move track boxes forward using their velocity
        pred_boxes_t = deepcopy(boxes_t)
        # TODO: update with precise time stamp
        pred_boxes_t[:, :2] += velo_t.detach() * age_t.unsqueeze(1) * 0.5 # s1 = s0 + v0 * delta_t

        # adj = velocity_adj((pred_boxes_t, boxes_d), (velo_t, velo_d))
        adj = class_velocity_adj((pred_boxes_t, boxes_d), (cls_t, cls_d))
        edge_index = adj_to_edge_index(adj)

        diff_box = boxes_d[edge_index[1, :], :] - boxes_t[edge_index[0, :], :]
        diff_time = age_t[edge_index[0, :]].unsqueeze(1)
        diff_position_pred = boxes_d[edge_index[1, :], :2] - pred_boxes_t[edge_index[0, :], :2]
        # edge_cost = cost[edge_index[0, :], edge_index[1, :]].unsqueeze(1)

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
