import os
import main as train_main

# ====== paths ======
VIS_PATH = "/mnt/f/Tri_modal_data/Temp_Tri_modal_data/vis_root"
IR_PATH  = "/mnt/f/Tri_modal_data/Temp_Tri_modal_data/ir_root"
SAR_PATH = "/mnt/f/Tri_modal_data/Temp_Tri_modal_data/sar_root"

COCO_PRETRAIN = "/mnt/d/Tri_modal_temp_code/TransVOD-master_yuanshi/pth/r50_deformable_detr_single_scale_dc5-checkpoint.pth"
# ===================


def run_singlebaseline():
    parser = train_main.get_args_parser()
    args = parser.parse_args([])  # do not read CLI

    # ====== set tri-modal single-frame (Stage1) ======
    args.dataset_file = "vid_single_3m"
    args.vid_path = VIS_PATH
    args.vid_path_ir = IR_PATH
    args.vid_path_sar = SAR_PATH

    # ====== align to configs/r50_train_single.sh ======
    args.epochs = 1
    args.batch_size = 1
    args.num_workers = 2

    args.num_feature_levels = 1   # must match single_scale weights
    args.num_queries = 300
    args.dilation = True          # must match dc5
    args.with_box_refine = True
    args.lr_drop_epochs = [6, 7]
    args.lr = 2e-4

    # classes
    args.num_classes = 3

    # pretrain loading
    args.coco_pretrain = True
    args.resume = COCO_PRETRAIN

    # output
    args.output_dir = "exps/singlebaseline/mydata_r50_singleScale_dc5_3m_stage1"
    os.makedirs(args.output_dir, exist_ok=True)

    # device
    args.device = "cuda"
    # ================================================

    # ====== debug prints (after args set) ======
    is_single = args.dataset_file in ["vid_single", "vid_single_3m"]
    print("ENGINE SELECT:", "single" if is_single else "multi")
    print("dataset_file =", args.dataset_file)
    print("num_ref_frames =", getattr(args, "num_ref_frames", None))
    # ==========================================

    train_main.main(args)


if __name__ == "__main__":
    run_singlebaseline()