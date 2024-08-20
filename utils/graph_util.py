# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from torch import Tensor
from copy import deepcopy


from torch_geometric.data import Batch
from torch_geometric.utils.unbatch import unbatch

from utils.data_util import BipartiteData

def adj_to_edge_index(adj):
    return torch.nonzero(adj).transpose(1, 0).long()

def class_velocity_adj(boxes, classes, time_diff=0.5):
    center_dist = torch.cdist(boxes[0][:, :2], boxes[1][:, :2], p=2.0)
    cls_mask = torch.eq(classes[0].unsqueeze(1), classes[1].unsqueeze(0))
    unknown_1 = torch.eq(classes[0].unsqueeze(1), 7)
    unknown_2 = torch.eq(classes[1].unsqueeze(0), 7)
    combined = unknown_1 | unknown_2
    cls_mask = cls_mask | combined
    cls_mask = torch.logical_not(cls_mask).float() * 1e16
    center_dist += cls_mask
    # size_diff = torch.cdist(boxes[0][:, 3:6],  boxes[1][:, 3:6], p=1.0)


    # max_velo = torch.tensor([4, 1, 3, 4, 5.5, 13, 3, 4], device=boxes[0].device)
    # distance_thresh = torch.gather(max_velo, 0, classes[0].long())
    # distance_thresh *= time_diff
    distance_thresh = 3
    # size_thresh = 0.8
    
    # Create masks based on the computed thresholds
    center_mask = torch.le(center_dist, distance_thresh).int()  # distance threshold
    # size_mask = torch.le(size_diff, size_thresh).int()

    adj = center_mask 
    return adj

def embedding_velocity_adj(boxes, embeddings, time_diff=0.5):
    center_dist = torch.cdist(boxes[0][:, :2], boxes[1][:, :2], p=2.0)
    embedding_dist = torch.cdist(embeddings[0], embeddings[1], p=2.0)
    center_dist += embedding_dist

    distance_thresh = 35

    adj = torch.le(center_dist, distance_thresh).int()
    return adj

def velocity_adj(boxes, dist_thresh, size_thresh, time_diff=0.5):

    track_boxes = boxes[0]
    detection_boxes = boxes[1]
    
    # dist_pct = 0.1
    # size_pct = 0.1
    # orientation_pct = 0.05

    # # Compute the thresholds based on the percentage differences
    # dist_thresh = 5 #torch.norm(delta_dist[:, :2], p=2.0, dim=1).unsqueeze(1) * dist_pct
    # size_thresh = 1 #torch.norm(track_boxes[:, 3:6], p=1.0, dim=1).unsqueeze(1) * size_pct
    # orientation_thresh = 0.5 #torch.abs(track_boxes[:, 6].unsqueeze(1)) * orientation_pct

    # Compute the differences
    center_dist = torch.cdist(track_boxes[:, :2], detection_boxes[:, :2], p=2.0)
    size_diff = torch.cdist(track_boxes[:, 3:6], detection_boxes[:, 3:6], p=1.0)
    # orientation_diff = torch.cdist(track_boxes[:, 6].unsqueeze(1), detection_boxes[:, 6].unsqueeze(1), p=1.0)

    # Create masks based on the computed thresholds
    center_mask = torch.le(center_dist, dist_thresh).int()  # distance threshold
    size_mask = torch.le(size_diff, size_thresh).int()  # size threshold
    # orientation_mask = torch.le(orientation_diff, orientation_thresh).int()  # orientation threshold
    
    # Combine all masks to create the final adjacency matrix
    adj = center_mask * size_mask #* orientation_mask
    
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
                      track_velo, track_age, det_batch, track_batch):

    det_boxes_list = unbatch(det_boxes, det_batch)
    det_class_list = unbatch(det_class, det_batch)

    track_boxes_list = unbatch(track_boxes, track_batch)
    track_class_list = unbatch(track_class, track_batch)
    track_velo_list = unbatch(track_velo, track_batch)

    track_age_list = unbatch(track_age, track_batch)

    data_list = []
    for boxes_d, cls_d, boxes_t, cls_t, age_t, velo_t in zip(det_boxes_list,
        det_class_list, track_boxes_list, track_class_list, track_age_list, track_velo_list):

        # Move track boxes forward using their velocity
        pred_boxes_t = deepcopy(boxes_t)
        # TODO: update with precise time stamp
        pred_boxes_t[:, :2] += velo_t.detach() * age_t.unsqueeze(1) * 0.5 # s1 = s0 + v0 * delta_t

        adj = class_velocity_adj((pred_boxes_t, boxes_d), (cls_t, cls_d))
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

def build_thresh_inter_graph(det_boxes, track_boxes, track_velo, track_age, 
                            det_batch, track_batch, dist_thresh, size_thresh):

    det_boxes_list = unbatch(det_boxes, det_batch)

    track_boxes_list = unbatch(track_boxes, track_batch)
    track_velo_list = unbatch(track_velo, track_batch)

    track_age_list = unbatch(track_age, track_batch)

    data_list = []
    for boxes_d, boxes_t, age_t, velo_t in zip(det_boxes_list,
        track_boxes_list, track_age_list, track_velo_list):

        # Move track boxes forward using their velocity
        pred_boxes_t = deepcopy(boxes_t)
        # TODO: update with precise time stamp
        delta_dist = velo_t.detach() * age_t.unsqueeze(1) * 0.5
        pred_boxes_t[:, :2] += delta_dist # s1 = s0 + v0 * delta_t

        adj = velocity_adj((pred_boxes_t, boxes_d), dist_thresh, size_thresh)
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