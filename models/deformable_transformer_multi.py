# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, n_temporal_decoder_layers=1,
                 num_ref_frames=3, fixed_pretrained_model=False, args=None):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.num_ref_frames = num_ref_frames
        self.two_stage_num_proposals = two_stage_num_proposals
        self.fixed_pretrained_model = fixed_pretrained_model
        self.n_temporal_query_layers = 3
        # self.TDAM = False
        self.TDAM = bool(getattr(args, "tdam", False))

        self.cqs_topk = 0 if args is None else int(getattr(args, "cqs_topk", 0))

        self.init_query_from_features = bool(getattr(args, "init_query_from_features", False))

        # Decoder fusion mode:
        # - 'gated'/'avg'/'concat': do per-modality cross-attn then fuse on query side
        # - 'msd': DAMSDet-style multispectral decoder: treat modalities as extra feature levels,
        #          run ONE MSDeformAttn so sampling offsets are predicted per (modality, level).
        # Use a separate arg name for Stage-2 to avoid changing historical default behavior.
        self.trimodal_fusion = 'msd'
        self.use_msd_decoder = True

        # Always use DAMSDet-style multispectral decoder for FINAL (temporal) decoding.
        # Hard-coded (no CLI switch).
        self.use_msd_temporal_decoder = True
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        dec_levels = num_feature_levels * 3 if self.use_msd_decoder else num_feature_levels
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          dec_levels, nhead, dec_n_points,
                                                          fusion=self.trimodal_fusion)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        # Temporal Transformer
        self.temporal_encoder_layer = TemporalDeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation,
            num_ref_frames, nhead, enc_n_points
        )

        self.temporal_query_layer1 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
        self.temporal_query_layer2 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
        self.temporal_query_layer3 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)



        # Final temporal decoder: hard-code DAMSDet-style multispectral decoding (modalities as extra levels).
        temporal_decoder_layer = DeformableTransformerDecoderLayer(
            d_model, dim_feedforward, dropout, activation,
            num_feature_levels * 3, nhead, dec_n_points,
            fusion='msd'
        )
        self.temporal_decoder = TemporalDeformableTransformerDecoder(
            temporal_decoder_layer, n_temporal_decoder_layers, False
        )

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        proposals = proposals.sigmoid() * scale
        pos = proposals[:, :, :, None] / dim_t
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    @staticmethod
    def _pack_msd_triplet(
        mem_vis: torch.Tensor,
        mem_ir: torch.Tensor,
        mem_sar: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        """Pack (VIS, IR, SAR) into a single multispectral sequence.

        This follows DAMSDet-style multispectral deformable attention:
        treat each modality's feature pyramid as extra feature levels.
        """
        # seq concat: [B, S, C] -> [B, 3S, C]
        src_cat = torch.cat([mem_vis, mem_ir, mem_sar], dim=1)

        # repeat spatial shapes: [L,2] -> [3L,2]
        spatial_shapes_msd = spatial_shapes.repeat(3, 1)

        # level_start_index: [L] -> [3L], with modality offsets
        # base_total = sum_l (H_l * W_l)
        base_total = spatial_shapes.prod(1).sum()
        level_start_index_msd = torch.cat(
            [level_start_index + i * base_total for i in range(3)], dim=0
        )

        # valid_ratios: [B,L,2] -> [B,3L,2]
        valid_ratios_msd = valid_ratios.repeat(1, 3, 1)

        # padding mask: [B,S] -> [B,3S]
        if padding_mask is None:
            padding_mask_msd = None
        else:
            padding_mask_msd = torch.cat([padding_mask, padding_mask, padding_mask], dim=1)

        return src_cat, spatial_shapes_msd, level_start_index_msd, valid_ratios_msd, padding_mask_msd

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, srcs_vis, srcs_ir, srcs_sar, masks, pos_embeds,
            query_embed=None, class_embed=None, warmup_alpha: Optional[float] = None):

        assert self.two_stage or query_embed is not None

        # --- tri-modal flatten ---
        src_flatten_vis = []
        src_flatten_ir = []
        src_flatten_sar = []

        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (src_v, src_i, src_s, mask, pos_embed) in enumerate(
                zip(srcs_vis, srcs_ir, srcs_sar, masks, pos_embeds)):

            # debug safety (可保留，稳定后删)
            # assert src_i.shape == src_v.shape and src_s.shape == src_v.shape, (src_v.shape, src_i.shape, src_s.shape)

            bs, c, h, w = src_v.shape
            spatial_shapes.append((h, w))

            src_v = src_v.flatten(2).transpose(1, 2)  # [bs, hw, c]
            src_i = src_i.flatten(2).transpose(1, 2)
            src_s = src_s.flatten(2).transpose(1, 2)

            mask_f = mask.flatten(1)  # [bs, hw]
            pos_f = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_f + self.level_embed[lvl].view(1, 1, -1)

            src_flatten_vis.append(src_v)
            src_flatten_ir.append(src_i)
            src_flatten_sar.append(src_s)

            mask_flatten.append(mask_f)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        src_flatten_vis = torch.cat(src_flatten_vis, 1)
        src_flatten_ir = torch.cat(src_flatten_ir, 1)
        src_flatten_sar = torch.cat(src_flatten_sar, 1)

        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten_vis.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        # --- end tri-modal flatten ---

        # encoder: 3 modalities
        memory_vis = self.encoder(src_flatten_vis, spatial_shapes, level_start_index, valid_ratios,
                                  lvl_pos_embed_flatten, mask_flatten)
        memory_ir = self.encoder(src_flatten_ir, spatial_shapes, level_start_index, valid_ratios,
                                 lvl_pos_embed_flatten, mask_flatten)
        memory_sar = self.encoder(src_flatten_sar, spatial_shapes, level_start_index, valid_ratios,
                                  lvl_pos_embed_flatten, mask_flatten)

        bs, _, c = memory_vis.shape

        if self.two_stage:
            # 1) 生成三模态 encoder proposals（two-stage）
            out_mem_vis, out_prop_vis = self.gen_encoder_output_proposals(memory_vis, mask_flatten, spatial_shapes)
            out_mem_ir,  out_prop_ir  = self.gen_encoder_output_proposals(memory_ir,  mask_flatten, spatial_shapes)
            out_mem_sar, out_prop_sar = self.gen_encoder_output_proposals(memory_sar, mask_flatten, spatial_shapes)

            # 2) 拼接后全局打分 Top-K（对齐 DAMSDet Competitive Query Selection 思路）
            out_mem_cat  = torch.cat([out_mem_vis, out_mem_ir, out_mem_sar], dim=1)
            out_prop_cat = torch.cat([out_prop_vis, out_prop_ir, out_prop_sar], dim=1)
            mask_cat     = torch.cat([mask_flatten, mask_flatten, mask_flatten], dim=1)
            valid_cat    = torch.isfinite(out_prop_cat).all(-1)

            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](out_mem_cat)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](out_mem_cat) + out_prop_cat

            # query-level score：max over classes（sigmoid 单调，不影响排序）
            scores = enc_outputs_class.sigmoid().max(-1)[0]
            scores = scores.masked_fill(mask_cat | (~valid_cat), float('-inf'))

            # 先取更大的候选池，再做二次筛选（便于后续替换更强的二次竞争评分）
            k_pool = min(self.two_stage_num_proposals * 3, scores.shape[1])
            topk_scores, topk_idx = torch.topk(scores, k_pool, dim=1)

            k_final = min(self.two_stage_num_proposals, k_pool)
            _, rel_idx = torch.topk(topk_scores, k_final, dim=1)
            final_idx = torch.gather(topk_idx, 1, rel_idx)

            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, final_idx.unsqueeze(-1).repeat(1, 1, 4)
            ).detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points

            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt_from_pos = torch.split(pos_trans_out, c, dim=2)

            # 可选：用被选中的 encoder token feature 初始化 tgt（更贴近 DAMSDet）
            if self.init_query_from_features:
                tgt = torch.gather(out_mem_cat, 1, final_idx.unsqueeze(-1).repeat(1, 1, c)).detach()
            else:
                tgt = tgt_from_pos

        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        # - fusion in {'avg','gated','concat'}: per-modality deformable cross-attn then fuse on query-side
        # - fusion == 'msd': DAMSDet-style multispectral deformable decoder (modalities as extra feature levels)
        if self.use_msd_decoder:
            src_cat, shapes_cat, lsi_cat, ratios_cat, mask_cat = self._pack_msd_triplet(
                memory_vis, memory_ir, memory_sar,
                spatial_shapes, level_start_index, valid_ratios,
                padding_mask=mask_flatten
            )
            hs, inter_references = self.decoder(
                tgt, reference_points, src_cat,
                shapes_cat, lsi_cat, ratios_cat,
                query_embed, mask_cat
            )
        else:
            memories = (memory_vis, memory_ir, memory_sar)
            hs, inter_references = self.decoder(
                tgt, reference_points, memories,
                spatial_shapes, level_start_index, valid_ratios,
                query_embed, mask_flatten
            )

        inter_references_out = inter_references
        # if self.two_stage:
        #     return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact

        if self.fixed_pretrained_model:
            # 旧代码里 detach 的是 memory/hs/inter_references；这里改成三路 memory 都 detach
            memory_vis = memory_vis.detach()
            memory_ir = memory_ir.detach()
            memory_sar = memory_sar.detach()
            hs = hs.detach()
            inter_references_out = inter_references_out.detach()

        # ------------------------------------------------------------------
        # Temporal Transformer
        # ------------------------------------------------------------------
        # ===== warmup gate: if alpha<=0, skip all temporal ops and make temporal outputs == static current-frame outputs =====
        if warmup_alpha is not None and warmup_alpha <= 1e-12:
            Kp1 = self.num_ref_frames + 1
            if bs % Kp1 != 0:
                raise ValueError(f"[multi-3m] dim0={bs} is not divisible by (num_ref_frames+1)={Kp1}. "
                                f"Check multi-frame input packaging.")
            B0 = bs // Kp1

            def _split_sample_major(x: torch.Tensor):
                x = x.contiguous().view(B0, Kp1, *x.shape[1:])
                return [x[:, t] for t in range(Kp1)]

            # static last layer outputs per-frame
            last_hs = hs[-1]  # [(K+1)*B, Q, C]
            last_reference_out = inter_references_out[-1]  # [(K+1)*B, Q, 2 or 4]
            last_hs_list = _split_sample_major(last_hs)
            last_reference_out_list = _split_sample_major(last_reference_out)

            # temporal outputs forced to current-frame static
            final_hs = last_hs_list[0]                 # [B, Q, C]
            final_references_out = last_reference_out_list[0]  # [B, Q, 2/4]

            # two-stage current-frame encoder outputs (if enabled)
            enc_outputs_class_cur = None
            enc_outputs_coord_unact_cur = None
            if self.two_stage:
                enc_outputs_class_cur = enc_outputs_class.contiguous().view(B0, Kp1, *enc_outputs_class.shape[1:])[:, 0]
                enc_outputs_coord_unact_cur = enc_outputs_coord_unact.contiguous().view(B0, Kp1, *enc_outputs_coord_unact.shape[1:])[:, 0]

            # current-frame decoder outputs for static heads
            hs_cur = hs.contiguous().view(hs.shape[0], B0, Kp1, *hs.shape[2:])[:, :, 0]
            inter_ref_cur = inter_references_out.contiguous().view(inter_references_out.shape[0], B0, Kp1, *inter_references_out.shape[2:])[:, :, 0]
            init_ref_cur = init_reference_out.contiguous().view(B0, Kp1, *init_reference_out.shape[1:])[:, 0]

            # optional: one-time debug
            if self.training and (not hasattr(self, "_dbg_warmup_skip_once")):
                self._dbg_warmup_skip_once = True
                print("[warmup-skip] warmup_alpha<=0, skip TDAM/TQE/temporal-decoder; temporal outputs = static current-frame")

            return (
                hs_cur,
                init_ref_cur,
                inter_ref_cur,
                enc_outputs_class_cur,
                enc_outputs_coord_unact_cur,
                final_hs,
                final_references_out
            )















        # IMPORTANT (bugfix):
        # The multi-frame collate function flattens (K+1) frames into dim0 in **sample-major** order:
        #   [s0_f0, s0_f1, ..., s0_fK, s1_f0, s1_f1, ..., s1_fK, ...]
        # Older code used torch.chunk(dim0, K+1) which assumes **time-major** order and breaks when batch_size>1.
        # Here we reshape by (B, K+1, ...) to robustly split current/ref frames for any batch_size.
        Kp1 = self.num_ref_frames + 1
        if bs % Kp1 != 0:
            raise ValueError(f"[multi-3m] dim0={bs} is not divisible by (num_ref_frames+1)={Kp1}. "
                             f"Check multi-frame input packaging.")
        B0 = bs // Kp1

        def _split_sample_major(x: torch.Tensor):
            """Split a tensor of shape [(B*(K+1)), ...] into a list length (K+1) of [B, ...] tensors."""
            x = x.contiguous().view(B0, Kp1, *x.shape[1:])
            return [x[:, t] for t in range(Kp1)]
        # 1) 构造当前帧三模态 memory tuple（给 temporal_decoder 用）
        memory_list_vis = _split_sample_major(memory_vis)
        memory_list_ir = _split_sample_major(memory_ir)
        memory_list_sar = _split_sample_major(memory_sar)

        cur_memory_vis = memory_list_vis[0]
        cur_memory_ir = memory_list_ir[0]
        cur_memory_sar = memory_list_sar[0]

        cur_memory = (cur_memory_vis, cur_memory_ir, cur_memory_sar)

        # 2) TDAM（temporal_encoder_layer）相关只在 TDAM=True 时计算，避免 TDAM=False 仍执行 ref_memory 等逻辑
        if self.TDAM:
            valid_ratios_cur = valid_ratios.contiguous().view(B0, Kp1, *valid_ratios.shape[1:])[:, 0]  # [B, L, 2]

            # print once
            if self.training and (not hasattr(self, "_dbg_tdam_once")):
                self._dbg_tdam_once = True
                print("[TDAM] enabled, num_ref_frames =", self.num_ref_frames,
                    "num_feature_levels =", int(spatial_shapes.shape[0]),
                    "spatial_shapes =", spatial_shapes.tolist())

            # capture-before flag (local)
            do_delta_check = self.training and (not hasattr(self, "_dbg_tdam_delta_once"))
            if do_delta_check:
                cur_mem_vis_before = cur_memory_vis.detach()
                cur_mem_ir_before  = cur_memory_ir.detach()
                cur_mem_sar_before = cur_memory_sar.detach()


            # 把 reference frames 当作 “levels”
            ref_spatial_shapes = spatial_shapes.expand(self.num_ref_frames, 2).contiguous()
            frame_start_index = torch.cat(
                (ref_spatial_shapes.new_zeros((1,)), ref_spatial_shapes.prod(1).cumsum(0)[:-1])
            ).contiguous()

            lvl_pos_list = _split_sample_major(lvl_pos_embed_flatten)
            cur_pos_embed = lvl_pos_list[0]                 # [B, sum_hw, C]
            ref_pos_embed = torch.cat(lvl_pos_list[1:], 1)  # [B, K*sum_hw, C]


            valid_ratios_ref = valid_ratios_cur.expand(valid_ratios_cur.shape[0], self.num_ref_frames, 2)


            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios_ref, device=cur_memory_vis.device
            )
            assert reference_points.shape[1] == cur_memory_vis.shape[1], (reference_points.shape, cur_memory_vis.shape)
            assert reference_points.shape[2] == self.num_ref_frames, reference_points.shape


            ref_memory_vis = torch.cat(memory_list_vis[1:], 1) + ref_pos_embed
            ref_memory_ir = torch.cat(memory_list_ir[1:], 1) + ref_pos_embed
            ref_memory_sar = torch.cat(memory_list_sar[1:], 1) + ref_pos_embed

            cur_memory_vis = self.temporal_encoder_layer(
                cur_memory_vis, cur_pos_embed, reference_points,
                ref_memory_vis, ref_spatial_shapes, frame_start_index
            )
            cur_memory_ir = self.temporal_encoder_layer(
                cur_memory_ir, cur_pos_embed, reference_points,
                ref_memory_ir, ref_spatial_shapes, frame_start_index
            )
            cur_memory_sar = self.temporal_encoder_layer(
                cur_memory_sar, cur_pos_embed, reference_points,
                ref_memory_sar, ref_spatial_shapes, frame_start_index
            )

            if do_delta_check:
                self._dbg_tdam_delta_once = True
                d_vis = (cur_memory_vis.detach() - cur_mem_vis_before).pow(2).mean().sqrt().item()
                d_ir  = (cur_memory_ir.detach()  - cur_mem_ir_before ).pow(2).mean().sqrt().item()
                d_sar = (cur_memory_sar.detach() - cur_mem_sar_before).pow(2).mean().sqrt().item()
                print("[TDAM-check] delta vis/ir/sar =", d_vis, d_ir, d_sar)


            cur_memory = (cur_memory_vis, cur_memory_ir, cur_memory_sar)


        # 3) temporal query enhancement (Top-K from ref_hs)
        last_hs = hs[-1]
        last_reference_out = inter_references_out[-1]

        last_hs_list = _split_sample_major(last_hs)
        last_reference_out_list = _split_sample_major(last_reference_out)

        # ---- Per-frame Competitive Query Selection (CQS) before TQE ----
        if (not self.two_stage) and (self.cqs_topk is not None) and (self.cqs_topk > 0):
            def _select_topk(hs_frame, ref_frame):
                logits = class_embed(hs_frame)              # [B, Q, num_classes]
                scores = logits.sigmoid().max(-1)[0]        # [B, Q]
                k = min(self.cqs_topk, hs_frame.shape[1])
                topk_idx = torch.topk(scores, k, dim=1)[1]  # [B, k]
                hs_sel = torch.gather(hs_frame, 1, topk_idx.unsqueeze(-1).repeat(1, 1, hs_frame.size(-1)))
                ref_sel = torch.gather(ref_frame, 1, topk_idx.unsqueeze(-1).repeat(1, 1, ref_frame.size(-1)))
                return hs_sel, ref_sel

            new_hs_list, new_ref_list = [], []
            for hsi, rfi in zip(last_hs_list, last_reference_out_list):
                hsi_sel, rfi_sel = _select_topk(hsi, rfi)
                new_hs_list.append(hsi_sel)
                new_ref_list.append(rfi_sel)

            last_hs_list = new_hs_list
            last_reference_out_list = new_ref_list
        # ---- end CQS ----

        # DEBUG (print once) — must be after CQS
        if self.training and (not hasattr(self, "_dbg_cqs_once")):
            self._dbg_cqs_once = True
            per_frame_lens = [x.shape[1] for x in last_hs_list]
            ref_len_dbg = sum(per_frame_lens[1:]) if len(per_frame_lens) > 1 else 0
            print("[CQS-check] cqs_topk =", self.cqs_topk,
                "per-frame Q len:", per_frame_lens,
                "ref_len:", ref_len_dbg,
                "num_ref_frames =", self.num_ref_frames)

        cur_hs = last_hs_list[0]
        ref_hs = torch.cat(last_hs_list[1:], 1)
        cur_reference_out = last_reference_out_list[0]

        ref_hs_logits = class_embed(ref_hs)     # [B, ref_len, num_cls]
        prob = ref_hs_logits.sigmoid()
        ref_len = ref_hs.shape[1]

        # 1) 用 “每个 query 的最大类别置信度” 作为 query-level score，避免同一 query 多类别重复入选
        scores = prob.max(-1)[0]               # [B, ref_len]

        # 2) 按比例采样（在 Top-K 集合内部）
        #    比例你可先固定为 0.8 / 0.5 / 0.3（对应旧的 80/50/30 思路）
        def _topk_ref_by_ratio(ratio: float):
            k = int(ref_len * ratio)
            k = max(1, min(ref_len, k))
            topk_idx = torch.topk(scores, k, dim=1)[1]  # [B, k]
            ref_in = torch.gather(ref_hs, 1, topk_idx.unsqueeze(-1).repeat(1, 1, ref_hs.size(-1)))  # [B,k,C]
            return k, topk_idx, ref_in

        k1, idx1, ref_hs_input1 = _topk_ref_by_ratio(0.8)
        if self.training and (not hasattr(self, "_dbg_topk_unique_once")):
            self._dbg_topk_unique_once = True
            uniq = torch.unique(idx1.reshape(-1)).numel()
            print("[TopK-unique] k1 =", k1, "unique =", uniq)
        cur_hs = self.temporal_query_layer1(cur_hs, ref_hs_input1)

        k2, idx2, ref_hs_input2 = _topk_ref_by_ratio(0.5)
        cur_hs = self.temporal_query_layer2(cur_hs, ref_hs_input2)

        k3, idx3, ref_hs_input3 = _topk_ref_by_ratio(0.3)
        cur_hs = self.temporal_query_layer3(cur_hs, ref_hs_input3)

        # debug print once (可保留一两次后删除)
        if self.training and (not hasattr(self, "_dbg_topk_once")):
            self._dbg_topk_once = True
            print("[TopK-check] ref_len =", ref_len,
                "k1/k2/k3 =", k1, k2, k3,
                "ratios =", (0.8, 0.5, 0.3))

        if self.training and (not hasattr(self, "_dbg_topk_once")):
            self._dbg_topk_once = True
            print("[TopK-check] ref_len =", ref_hs.shape[1],
                "k1/k2/k3 =", k1, k2, k3,
                "expected =", (80*self.num_ref_frames, 50*self.num_ref_frames, 30*self.num_ref_frames))


        # 4) temporal decoder: 输入三模态当前帧 memory tuple
        # 注意：这里必须用 “原始 valid_ratios[0:1]” (shape [1, n_levels, 2])，不要用 TDAM 的 valid_ratios_ref
        valid_ratios_cur = valid_ratios.contiguous().view(B0, Kp1, *valid_ratios.shape[1:])[:, 0]
        if self.use_msd_temporal_decoder:
            cur_src_cat, cur_shapes_cat, cur_lsi_cat, cur_ratios_cat, _ = self._pack_msd_triplet(
                cur_memory_vis, cur_memory_ir, cur_memory_sar,
                spatial_shapes, level_start_index, valid_ratios_cur,
                padding_mask=None
            )
            final_hs, final_references_out = self.temporal_decoder(
                cur_hs, cur_reference_out, cur_src_cat,
                cur_shapes_cat, cur_lsi_cat, cur_ratios_cat,
                None, None
            )
        else:
            final_hs, final_references_out = self.temporal_decoder(
                cur_hs, cur_reference_out, cur_memory,
                spatial_shapes, level_start_index, valid_ratios_cur,
                None, None
            )
        if self.training and (not hasattr(self, "_dbg_td_once")):
            self._dbg_td_once = True
            # final_hs: [n_layers, B, Q, C] or [1,B,Q,C] depending implementation
            # use last layer output
            td_out = final_hs[-1] if final_hs.dim() == 4 else final_hs
            delta_q = (td_out.detach() - cur_hs.detach()).pow(2).mean().sqrt().item()
            print("[TD-check] delta(final_hs vs cur_hs) =", delta_q)


        # --- select current-frame encoder outputs for two-stage ---
        enc_outputs_class_cur = None
        enc_outputs_coord_unact_cur = None
        if self.two_stage:
            # enc_outputs_*: [(K+1)*B, S, ...]  -> chunk into (K+1) pieces, take current frame [0]
            enc_outputs_class_cur = enc_outputs_class.contiguous().view(B0, Kp1, *enc_outputs_class.shape[1:])[:, 0]
            enc_outputs_coord_unact_cur = enc_outputs_coord_unact.contiguous().view(B0, Kp1, *enc_outputs_coord_unact.shape[1:])[:, 0]
        # --- end ---

        # Return only current-frame outputs for loss/heads. Keep batch dimension = B0 (not forced to 1).
        hs_cur = hs.contiguous().view(hs.shape[0], B0, Kp1, *hs.shape[2:])[:, :, 0]
        inter_ref_cur = inter_references_out.contiguous().view(inter_references_out.shape[0], B0, Kp1, *inter_references_out.shape[2:])[:, :, 0]
        init_ref_cur = init_reference_out.contiguous().view(B0, Kp1, *init_reference_out.shape[1:])[:, 0]

        return (
            hs_cur,
            init_ref_cur,
            inter_ref_cur,
            enc_outputs_class_cur,
            enc_outputs_coord_unact_cur,
            final_hs,
            final_references_out
        )



class TemporalQueryEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, query, ref_query, query_pos=None, ref_query_pos=None):
        q = k = self.with_pos_embed(query, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), query.transpose(0, 1))[0].transpose(0, 1)
        tgt = query + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos).transpose(0, 1),
            self.with_pos_embed(ref_query, ref_query_pos).transpose(0, 1),
            ref_query.transpose(0, 1)
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt = self.forward_ffn(tgt)
        return tgt


class TemporalQueryEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, query, ref_query, query_pos=None, ref_query_pos=None):
        output = query
        for _, layer in enumerate(self.layers):
            output = layer(output, ref_query, query_pos, ref_query_pos)
        return output


class TemporalDeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1,
                 activation='relu', num_ref_frames=3, n_heads=8, n_points=4):
        super().__init__()

        self.cross_attn = MSDeformAttn(d_model, num_ref_frames, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, frame_start_index, src_padding_mask=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, frame_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt = self.forward_ffn(tgt)
        return tgt


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src = self.forward_ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 fusion: str = 'gated'):
        super().__init__()

        self.fusion = fusion

        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # query-side fusion params (only used when fusion != 'msd')
        if fusion == 'gated':
            self.modal_gate = nn.Linear(d_model, 3)
        elif fusion == 'concat':
            self.fuse = nn.Linear(d_model * 3, d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if isinstance(src, (tuple, list)):
            # fusion on query-side (one deformable cross-attn per modality)
            src_vis, src_ir, src_sar = src

            q_in = self.with_pos_embed(tgt, query_pos)
            out_vis = self.cross_attn(q_in, reference_points, src_vis, src_spatial_shapes, level_start_index, src_padding_mask)
            out_ir  = self.cross_attn(q_in, reference_points, src_ir,  src_spatial_shapes, level_start_index, src_padding_mask)
            out_sar = self.cross_attn(q_in, reference_points, src_sar, src_spatial_shapes, level_start_index, src_padding_mask)

            if self.fusion == 'avg':
                tgt2 = (out_vis + out_ir + out_sar) / 3.0
            elif self.fusion == 'gated':
                gate = torch.softmax(self.modal_gate(tgt), dim=-1)  # [B, Q, 3]
                tgt2 = gate[..., 0:1] * out_vis + gate[..., 1:2] * out_ir + gate[..., 2:3] * out_sar
            elif self.fusion == 'concat':
                tgt2 = self.fuse(torch.cat([out_vis, out_ir, out_sar], dim=-1))
            else:
                raise ValueError(f"Unknown fusion mode for tuple src: {self.fusion}")
        else:
            # DAMSDet-style multispectral deformable decoder (fusion=='msd')
            # or single-modality path (src is already packed)
            tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                                   reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt = self.forward_ffn(tgt)
        return tgt


class TemporalDeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            output = layer(output, query_pos, reference_points_input, src,
                           src_spatial_shapes, src_level_start_index, src_padding_mask)

            self.bbox_embed = None
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            output = layer(output, query_pos, reference_points_input, src,
                           src_spatial_shapes, src_level_start_index, src_padding_mask)

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        n_temporal_decoder_layers=args.n_temporal_decoder_layers,
        num_ref_frames=args.num_ref_frames,
        fixed_pretrained_model=args.fixed_pretrained_model,
        args=args
    )
