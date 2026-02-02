import os
import main as train_main

VIS_PATH = "/mnt/f/Tri_modal_data/Temp_Tri_modal_data/vis_root"
IR_PATH  = "/mnt/f/Tri_modal_data/Temp_Tri_modal_data/ir_root"
SAR_PATH = "/mnt/f/Tri_modal_data/Temp_Tri_modal_data/sar_root"

SINGLE_CKPT = "/mnt/d/Tri_modal_temp_code/TransVOD-master_yuanshi/pth/single_checkpoint0043.pth"

def run_multi():
    parser = train_main.get_args_parser()
    args = parser.parse_args([])

    # ===== 关键：三模态多帧 =====
    args.dataset_file = "vid_multi_3m"

    # ===== 对齐 r50_train_multi.sh（建议先验证用小配置）=====
    args.backbone = "resnet50"
    args.epochs = 1          # 验证阶段建议 1
    args.num_feature_levels = 1
    args.num_queries = 300
    args.dilation = True
    args.batch_size = 1
    args.num_ref_frames = 3
    args.cqs_topk = 50   # 例如每帧筛到 100
    args.lr_drop_epochs = [4, 6]
    args.num_workers = 0     # 验证阶段建议 0
    args.with_box_refine = True
    args.lr = 2e-4

    # ===== 三路数据根目录分别设置 =====
    args.vid_path = VIS_PATH
    args.vid_path_ir = IR_PATH
    args.vid_path_sar = SAR_PATH
    args.num_classes = 3

    # ===== 从单帧初始化 =====
    args.coco_pretrain = False
    args.resume = SINGLE_CKPT

    # ===== 验证阶段：只跑 eval 更快（可保留）=====
    args.eval = False

    # ===== 输出 =====
    args.output_dir = "exps/_debug_multi3m_collate"
    os.makedirs(args.output_dir, exist_ok=True)

    # ===== 设备 =====
    args.device = "cuda"

    train_main.main(args)

if __name__ == "__main__":
    run_multi()
