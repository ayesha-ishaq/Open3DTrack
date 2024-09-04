# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------

import torch.nn as nn
import copy
import torch

from base import BaseModel
from model.transformers import FFN, TransformerEncoderLayer, TransformerDecoderLayer

class STTransformerModel(BaseModel):
    def __init__(self, d_model, nhead, dropout, encoder_nlayers, decoder_nlayers, 
                       norm_first, cross_attn_value_gate, in_channel, out_channel, num_classes):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.dropout_p = dropout
        self.encoder_nlayers = encoder_nlayers
        self.decoder_nlayers = decoder_nlayers
        self.norm_first = norm_first
        self.num_classes = num_classes
        self.edge_attr_cross_attn = True
        self.cross_attn_value_gate = cross_attn_value_gate

        # self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        # self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        # self.relu = nn.Tanh()

        self.embedding = nn.Linear(9, self.d_model)
        self.embedding_edge = nn.Linear(10, self.d_model)

        # self.embedding_score = nn.Linear(11, self.d_model)

        spatial_encoder_layer = TransformerEncoderLayer(self.d_model, self.nhead, self.dropout_p,
                                                        norm_first=self.norm_first)
        decoder_layer = TransformerDecoderLayer(self.d_model, self.nhead, self.dropout_p,
                                                norm_first=self.norm_first,
                                                edge_attr_cross_attn=self.edge_attr_cross_attn,
                                                cross_attn_value_gate=self.cross_attn_value_gate)

        self.spatial_encoders = _get_clones(spatial_encoder_layer, self.encoder_nlayers)
        self.decoders = _get_clones(decoder_layer, self.decoder_nlayers)

        if self.norm_first:
            self.norm_final = nn.LayerNorm(self.d_model)
            self.norm_final_edge = nn.LayerNorm(self.d_model)

        self.affinity = FFN(self.d_model)
        self.velocity = FFN(self.d_model, 2)
        # self.score = FFN(self.d_model)
        # self.sigmoid = nn.Sigmoid()

    
    def forward(self, dets_in, #dets_pts, dets_pt_fts, 
                tracks_in, #trk_pts, trk_pt_fts, 
                edge_index_det, edge_index_track,
                edge_index_inter, edge_attr_inter=None):

        if tracks_in.size(1) != self.d_model:
            # trk_emb = self._pointpillars_block(trk_pts, trk_pt_fts)
            # tracks = self.embedding(torch.cat([tracks_in, trk_emb.squeeze(dim=-1)], -1)) 
            tracks = self.embedding(tracks_in)

        else:
            tracks = tracks_in
        
        # det_emb = self._pointpillars_block(dets_pts, dets_pt_fts)
        # dets = self.embedding(torch.cat([dets_in, det_emb.squeeze(dim=-1)], -1)) 
        dets = self.embedding(dets_in)
        # CHANGE TRAIN THRESHOLDS
        edge_attr_inter = self.embedding_edge(edge_attr_inter)

        # Transformer encoder and decoder
        tracks = self._enc_block(tracks, edge_index_track)
        dets, edge_attr_inter, attn_weights = self._dec_block(
            tracks, dets, edge_index_inter, edge_index_det, edge_attr_inter)
        
        if self.norm_first:
            dets = self.norm_final(dets)
            edge_attr_inter = self.norm_final_edge(edge_attr_inter)

        affinity = [self.affinity(edge_attr_inter)]
        pred_velo = self.velocity(dets)
        # score_dets = self.sigmoid(self.score(dets))
        
        return affinity, tracks, dets, pred_velo#, score_dets

    def _pointpillars_block(self, det_pts, det_features):
        
        non_zero_mask = (det_pts != 0).any(dim=-1, keepdim=True) 
        det_nbr_points = torch.sum(non_zero_mask.squeeze(dim=-1), dim=-1)
        det_sum = torch.sum(det_pts, dim=1, keepdim=True) / det_nbr_points[:, None, None] 
        offset_det_center = torch.where(non_zero_mask, det_pts - det_sum, det_pts)
        x = torch.cat([offset_det_center, det_features], dim=-1) 
        x = x.permute(0, 2, 1).contiguous()
        x = self.relu(self.bn(self.conv(x)))
        x = torch.max(x, dim=-1)[0]
       
        return x

    def _enc_block(self, x, edge_index):
        for spt_layer in self.spatial_encoders:
            x = spt_layer(x, edge_index)
        return x
    
    def _dec_block(self, src, tgt, edge_index_inter, edge_index_tgt, edge_attr_inter=None):
        attn_weights = []
        # tracks_out = []
        for layer in self.decoders:
            tgt, edge_attr_inter, attn = layer(src, tgt, edge_index_inter, edge_index_tgt, edge_attr_inter)
            # tracks_out.append(tracks)
            attn_weights.append(attn[1])
        return tgt, edge_attr_inter, attn_weights
    

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])




