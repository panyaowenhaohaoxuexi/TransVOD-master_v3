import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets

import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset

from models import build_single, build_multi


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr_drop', default=5, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    parser.add_argument('--num_ref_frames', default=3, type=int, help='number of reference frames')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    parser.add_argument('--cqs_topk', default=0, type=int,
                        help='per-frame Competitive Query Selection topk before TQE; 0 disables')

    # tri-modal
    parser.add_argument('--trimodal_decoder', default=False, action='store_true',
                        help='use tri-modal query-fusion decoder for (VIS/IR/SAR)')
    parser.add_argument('--trimodal_fusion', default='avg', type=str,
                        choices=['avg', 'gated', 'concat', 'msd'],
                        help='query-side fusion type; set to msd to use DAMSDet-style multispectral deformable decoder (treat modalities as extra feature levels)')
    parser.add_argument('--trimodal_fusion_multi', default='gated', type=str,
                        choices=['avg', 'gated', 'concat', 'msd'],
                        help='fusion type for multi-frame (Stage-2) transformer decoder; default keeps historical behavior (gated). Set to msd to use DAMSDet-style multispectral deformable decoder.')
    parser.add_argument('--init_query_from_features', default=False, action='store_true',
                        help='init decoder tgt from topk encoder tokens (CQS-style)')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--n_temporal_decoder_layers', default=1, type=int)
    parser.add_argument('--interval1', default=20, type=int)
    parser.add_argument('--interval2', default=60, type=int)

    parser.add_argument("--fixed_pretrained_model", default=False, action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='vid_multi')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--vid_path', default='./data/vid', type=str)

    # tri-modal roots (scheme A)
    parser.add_argument('--vid_path_ir', default='', type=str,
                        help='IR root dir (scheme A). If empty, use --vid_path.')
    parser.add_argument('--vid_path_sar', default='', type=str,
                        help='SAR root dir (scheme A). If empty, use --vid_path.')

    parser.add_argument('--coco_pretrain', default=False, action='store_true')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # Stage-2 warmup (multi-frame): mix static and temporal outputs early to avoid degrading a strong Stage-1 baseline.
    parser.add_argument('--warmup_enable', action='store_true',
                        help='enable warmup in Stage-2 training (mix static/temporal outputs)')
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help='number of warmup epochs for output-mix warmup (mode-1)')
    parser.add_argument('--warmup_schedule', default='linear', type=str,
                        choices=['linear', 'cos'],
                        help='alpha schedule during warmup (linear or cosine)')

    # Stage-2 progressive unfreeze (Stage-C): unfreeze spatial decoder last layers for higher upper bound
    parser.add_argument('--unfreeze_decoder_last_n', default=0, type=int,
                        help='unfreeze last N layers of spatial decoder in Stage-2 (0 disables)')
    parser.add_argument('--unfreeze_decoder_start_epoch', default=-1, type=int,
                        help='epoch to start unfreezing decoder last layers; -1 means warmup_epochs')
    parser.add_argument('--lr_decoder', default=1e-5, type=float,
                        help='learning rate for unfrozen spatial decoder layers')

    return parser


def main(args):
    print(args.dataset_file, 11111111)

    # ===== single/multi 判定（vid_single_3m 也算 single） =====
    is_single = args.dataset_file in ["vid_single", "vid_single_3m"]

    # choose engine/utils
    if is_single:
        from engine_single import evaluate, train_one_epoch
        import util.misc as utils
    else:
        from engine_multi import evaluate, train_one_epoch
        if args.dataset_file == "vid_multi_3m":
            import util.misc_multi_3m as utils   # 三模态多帧 split(9)
        else:
            import util.misc_multi as utils      # 原来的单模态 multi

    device = torch.device(args.device)
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ===== 显式按 is_single 构建模型 =====
    if is_single:
        model, criterion, postprocessors = build_single(args)
    else:
        model, criterion, postprocessors = build_multi(args)

    model.to(device)
    model_without_ddp = model

    print("[where-am-i] model class:", model_without_ddp.__class__)
    print("[where-am-i] transformer class:", model_without_ddp.transformer.__class__)
    print("[where-am-i] transformer module:", model_without_ddp.transformer.__class__.__module__)

    # ===== dataset =====
    dataset_train = build_dataset(image_set='train_vid', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    def match_name_keywords(n, name_keywords):
        for b in name_keywords:
            if b in n:
                return True
        return False

    # ===== stage-C helpers =====
    def build_optimizer_and_scheduler():
        # If lr_drop_epochs is None, fall back to [lr_drop]
        milestones = args.lr_drop_epochs
        if milestones is None:
            milestones = [int(args.lr_drop)]
        print("[lr] milestones:", milestones)

        param_dicts = [
            {
                "params": [
                    p for n, p in model_without_ddp.named_parameters()
                    if not match_name_keywords(n, args.lr_backbone_names)
                    and not match_name_keywords(n, args.lr_linear_proj_names)
                    and p.requires_grad
                ],
                "lr": args.lr,
            },
            {
                "params": [
                    p for n, p in model_without_ddp.named_parameters()
                    if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
                ],
                "lr": args.lr_backbone,
            },
            {
                "params": [
                    p for n, p in model_without_ddp.named_parameters()
                    if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad
                ],
                "lr": args.lr * args.lr_linear_proj_mult,
            }
        ]

        if args.sgd:
            opt = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        else:
            opt = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones)
        return opt, sch, milestones

    def unfreeze_spatial_decoder_last_layers(n_last: int):
        """
        Unfreeze last n_last layers of spatial decoder:
        names start with 'transformer.decoder.layers.{i}.'
        """
        if n_last <= 0:
            return [], None

        dec = model_without_ddp.transformer.decoder
        num_layers = int(getattr(dec, "num_layers", 0))
        if num_layers <= 0:
            num_layers = len(dec.layers)

        start = max(0, num_layers - int(n_last))
        end = num_layers - 1
        prefixes = tuple([f"transformer.decoder.layers.{i}." for i in range(start, num_layers)])

        new_params = []
        new_names = []
        for name, p in model_without_ddp.named_parameters():
            if name.startswith(prefixes):
                if not p.requires_grad:
                    p.requires_grad = True
                    new_params.append(p)
                    new_names.append(name)

        return new_params, (start, end), new_names

    # ===== DDP wrap (keep same as your original flow) =====
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # ===== base ds for eval =====
    if args.dataset_file == "coco_panoptic":
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    # ===== resume / load =====
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        ckpt_sd_raw = checkpoint.get("model", checkpoint)

        # ---- eval: load everything possible (key in model + shape match) ----
        if args.eval:
            model_sd_now = model_without_ddp.state_dict()
            eval_sd = {k: v for k, v in ckpt_sd_raw.items()
                       if (k in model_sd_now) and (model_sd_now[k].shape == v.shape)}
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(eval_sd, strict=False)

        # ---- train ----
        else:
            # freeze non-temporal params only when doing temporal finetune stage (coco_pretrain==False)
            if not args.coco_pretrain:
                for name, param in model_without_ddp.named_parameters():
                    if ("temporal" in name) or name.startswith("temp_"):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

                trainable = [n for n, p in model_without_ddp.named_parameters() if p.requires_grad]
                frozen = [n for n, p in model_without_ddp.named_parameters() if not p.requires_grad]
                print("[freeze-check] trainable:", len(trainable), "frozen:", len(frozen))
                print("[freeze-check] first trainable:", trainable[:20])

            # ---------- load by (key in model) AND (shape match) ----------
            model_sd = model_without_ddp.state_dict()
            load_sd = {k: v for k, v in ckpt_sd_raw.items()
                       if (k in model_sd) and (tuple(model_sd[k].shape) == tuple(v.shape))}

            # ===== debug: resume coverage =====
            in_model = [k for k in ckpt_sd_raw.keys() if k in model_sd]
            not_in_model = [k for k in ckpt_sd_raw.keys() if k not in model_sd]
            shape_mismatch = [k for k in in_model if tuple(ckpt_sd_raw[k].shape) != tuple(model_sd[k].shape)]
            loadable = list(load_sd.keys())

            print("====== [resume-debug] ckpt keys:", len(ckpt_sd_raw))
            print("====== [resume-debug] model keys:", len(model_sd))
            print("====== [resume-debug] ckpt keys in model:", len(in_model))
            print("====== [resume-debug] ckpt keys NOT in model:", len(not_in_model))
            print("====== [resume-debug] shape mismatch:", len(shape_mismatch))
            print("====== [resume-debug] loadable(after key+shape):", len(loadable))

            critical_prefix = ["backbone", "transformer", "input_proj", "bbox_embed", "class_embed", "temp"]
            for pfx in critical_prefix:
                c_in = sum(k.startswith(pfx) for k in ckpt_sd_raw.keys())
                c_load = sum(k.startswith(pfx) for k in loadable)
                print(f"====== [resume-debug] prefix={pfx:12s} ckpt={c_in:4d} loadable={c_load:4d}")

            print("====== [resume-debug] examples NOT in model:", not_in_model[:20])
            print("====== [resume-debug] examples shape mismatch:", shape_mismatch[:20])

            denom = len([k for k in ckpt_sd_raw.keys() if k in model_sd])
            if denom > 0 and len(loadable) < 0.7 * denom:
                raise RuntimeError(
                    "Resume coverage too low: most ckpt params are not loaded. "
                    "Check key names / shapes."
                )

            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(load_sd, strict=False)

            # ===== warm-start temp heads from Stage-1 heads when resuming single->multi =====
            # IMPORTANT: when two_stage=True, class_embed/bbox_embed has (dec_layers+1) heads;
            # the last one (-1) is for encoder proposals. We should copy from the LAST DECODER layer head.
            if (not args.coco_pretrain) and hasattr(model_without_ddp, "temp_class_embed"):
                import torch.nn as nn
                need = (missing_keys is not None) and any(
                    k.startswith("temp_class_embed") or k.startswith("temp_bbox_embed")
                    for k in missing_keys
                )
                if need:
                    with torch.no_grad():
                        last_dec_id = model_without_ddp.transformer.decoder.num_layers - 1

                        # class head (use decoder last layer head)
                        if isinstance(model_without_ddp.class_embed, nn.ModuleList):
                            src_cls = model_without_ddp.class_embed[last_dec_id]
                        else:
                            src_cls = model_without_ddp.class_embed
                        model_without_ddp.temp_class_embed.weight.copy_(src_cls.weight)
                        model_without_ddp.temp_class_embed.bias.copy_(src_cls.bias)

                        # box head (use decoder last layer head)
                        if isinstance(model_without_ddp.bbox_embed, nn.ModuleList):
                            src_box = model_without_ddp.bbox_embed[last_dec_id]
                        else:
                            src_box = model_without_ddp.bbox_embed
                        model_without_ddp.temp_bbox_embed.load_state_dict(src_box.state_dict())

                    print("[warmstart] copied decoder(last) class_embed/bbox_embed -> temp_*")

        unexpected_keys = [
            k for k in unexpected_keys
            if not (k.endswith('total_params') or k.endswith('total_ops'))
        ]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

    # ===== eval-only =====
    if args.eval:
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            args=args, epoch=args.start_epoch
        )
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    # ===== build optimizer/scheduler AFTER resume+freeze (critical for Stage-2 finetune) =====
    optimizer, lr_scheduler, milestones = build_optimizer_and_scheduler()

    # ===== optimizer check =====
    opt_params = []
    for g in optimizer.param_groups:
        opt_params += list(g["params"])
    opt_params_set = set([id(p) for p in opt_params])
    trainable_params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    trainable_set = set([id(p) for p in trainable_params])
    print("[opt-check] optimizer params:", len(opt_params))
    print("[opt-check] trainable params:", len(trainable_params))
    print("[opt-check] optimizer==trainable:", opt_params_set == trainable_set)

    # total/trainable params log
    n_total = sum(p.numel() for p in model_without_ddp.parameters())
    n_trainable = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print("number of params (total):", n_total)
    print("number of params (trainable):", n_trainable)

    # ===== training with per-epoch validation + best checkpoint + delete non-best epoch ckpt =====
    print("Start training")
    start_time = time.time()

    best_metric = -1.0
    best_epoch = -1
    best_epoch_ckpt_path = None

    stage_c_done = False

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # ===== Stage-C: unfreeze spatial decoder last N layers (after warmup by default) =====
        if (not is_single) and (not args.coco_pretrain) and (int(getattr(args, "unfreeze_decoder_last_n", 0)) > 0):
            start_ep = int(getattr(args, "unfreeze_decoder_start_epoch", -1))
            if start_ep < 0:
                start_ep = int(getattr(args, "warmup_epochs", 0))

            if (not stage_c_done) and (epoch >= start_ep):
                n_last = int(getattr(args, "unfreeze_decoder_last_n", 0))
                lr_dec = float(getattr(args, "lr_decoder", 1e-5))

                new_params, lrng, new_names = unfreeze_spatial_decoder_last_layers(n_last)

                if len(new_params) > 0:
                    optimizer.add_param_group({
                        "params": new_params,
                        "lr": lr_dec,
                        "weight_decay": args.weight_decay,
                    })

                    # === FIX: make MultiStepLR happy when last_epoch != -1 ===
                    for pg in optimizer.param_groups:
                        if "initial_lr" not in pg:
                            pg["initial_lr"] = pg["lr"]
                   
                    # Rebuild scheduler because MultiStepLR stores base_lrs per param_group at init
                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer, milestones, last_epoch=epoch - 1
                    )

                stage_c_done = True
                print(f"[stageC] start_epoch={start_ep}, unfreeze_last_n={n_last}, "
                      f"layers={lrng}, lr_decoder={lr_dec}, new_params={len(new_params)}")
                if len(new_names) > 0:
                    print("[stageC] example unfrozen names:", new_names[:10])

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args
        )
        lr_scheduler.step()

        print('args.output_dir', args.output_dir)

        # ---- save regular checkpoints ----
        epoch_ckpt_path = None
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, output_dir / 'checkpoint.pth')

            epoch_ckpt_path = output_dir / f'checkpoint{epoch:04}.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, epoch_ckpt_path)

        # ---- validate each epoch ----
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            args=args, epoch=epoch
        )

        # ---- extract mAP50:95 and mAP50 ----
        cur_ap5095 = None
        cur_ap50 = None
        if isinstance(test_stats, dict) and ('coco_eval_bbox' in test_stats):
            try:
                cur_ap5095 = float(test_stats['coco_eval_bbox'][0])  # mAP50:95
                cur_ap50 = float(test_stats['coco_eval_bbox'][1])    # mAP50
            except Exception:
                cur_ap5095, cur_ap50 = None, None

        # ---- choose best by mAP50 ----
        key_metric = cur_ap50
        is_best = False

        if (key_metric is not None) and args.output_dir and utils.is_main_process() and (key_metric > best_metric):
            is_best = True

            if best_epoch_ckpt_path is not None and best_epoch_ckpt_path.exists():
                try:
                    best_epoch_ckpt_path.unlink()
                except Exception:
                    pass

            best_metric = key_metric
            best_epoch = epoch
            best_epoch_ckpt_path = epoch_ckpt_path

            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_metric_name': 'mAP50',
                'best_metric': best_metric,
                'best_epoch': best_epoch,
                'best_map5095': cur_ap5095,
                'best_map50': cur_ap50,
                'args': args,
            }, output_dir / 'best.pth')

            try:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "best_eval.pth")
            except Exception:
                pass

        # ---- delete non-best epoch checkpoint ----
        if args.output_dir and utils.is_main_process():
            if (epoch_ckpt_path is not None) and (not is_best) and epoch_ckpt_path.exists():
                try:
                    epoch_ckpt_path.unlink()
                except Exception:
                    pass

        # ---- log train + val metrics into log.txt ----
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in (test_stats.items() if isinstance(test_stats, dict) else [])},
            'val_map50': cur_ap50,
            'val_map5095': cur_ap5095,
            'epoch': epoch,
            'n_parameters_total': n_total,
            'n_parameters_trainable': n_trainable,
            'best_metric_name': 'mAP50',
            'best_metric': best_metric,
            'best_epoch': best_epoch,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
