
import os
import main as train_main

VIS_PATH = "/mnt/f/Tri_modal_data/Temp_Tri_modal_data/vis_root"
IR_PATH  = "/mnt/f/Tri_modal_data/Temp_Tri_modal_data/ir_root"
SAR_PATH = "/mnt/f/Tri_modal_data/Temp_Tri_modal_data/sar_root"

SINGLE_CKPT = "/mnt/d/Tri_modal_temp_code/TransVOD_model_pth/Tri_modal_single/best.pth"

# -------- Stage-2 warmup (mode-1: output mixing) --------
USE_WARMUP = True
WARMUP_EPOCHS = 5
WARMUP_SCHEDULE = "linear"  # {linear, cos}

def run_multi():
    parser = train_main.get_args_parser()
    args = parser.parse_args([])

    # ===== tri-modal multi-frame (Stage-2) =====
    args.dataset_file = "vid_multi_3m"

    # ===== align Stage-1 structure params =====
    args.backbone = "resnet50"
    args.epochs = 1
    args.num_feature_levels = 1
    args.num_queries = 300
    args.dilation = True
    args.batch_size = 1
    args.num_ref_frames = 3
    args.cqs_topk = 0
    args.two_stage = True
    args.tdam = True
    args.lr_drop_epochs = [30, 40]
    args.num_workers = 0
    args.with_box_refine = True
    args.lr = 2e-4

    # tri-modal decoder / fusion (keep identical to Stage-1)
    args.trimodal_decoder = True
    args.trimodal_fusion = "msd"
    args.trimodal_fusion_multi = "msd"
    args.init_query_from_features = True

    # ===== warmup =====
    args.warmup_enable = USE_WARMUP
    args.warmup_epochs = WARMUP_EPOCHS
    args.warmup_schedule = WARMUP_SCHEDULE

    # ===== Stage-D/E: progressive unfreeze for higher upper bound =====
    # Stage-D: unfreeze input_proj (shared projection convs)
    args.unfreeze_input_proj = True
    args.unfreeze_input_proj_start_epoch = args.warmup_epochs  # after warmup by default
    args.lr_input_proj = 2e-5

    # Stage-E: unfreeze encoder last 1 layer (spatial encoder)
    args.unfreeze_encoder_last_n = 1
    args.unfreeze_encoder_start_epoch = args.warmup_epochs + 5  # delay for stability
    args.lr_encoder = 2e-5

    # (optional) Stage-C decoder unfreeze (disable here; set >0 if you also want C)
    args.unfreeze_decoder_last_n = 0
    args.unfreeze_decoder_start_epoch = args.warmup_epochs
    args.lr_decoder = 1e-5

    # ===== data roots =====
    args.vid_path = VIS_PATH
    args.vid_path_ir = IR_PATH
    args.vid_path_sar = SAR_PATH
    args.num_classes = 3

    # ===== init from Stage-1 =====
    args.coco_pretrain = False
    args.resume = SINGLE_CKPT

    # ===== train =====
    args.eval = False

    # ===== output =====
    args.output_dir = "exps/multibaseline/v6_mydata_stage2_warmup_DE"
    os.makedirs(args.output_dir, exist_ok=True)

    args.device = "cuda"

    train_main.main(args)

if __name__ == "__main__":
    run_multi()
