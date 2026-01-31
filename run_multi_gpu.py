import os
import main as train_main

VID_PATH = "/mnt/f/Tri_modal_data/diyiqi_honwai_augment_1C"
SINGLE_CKPT = "/mnt/d/Tri_modal_temp_code/TransVOD-master_yuanshi/pth/single_checkpoint0043.pth"

def run_multi():
    parser = train_main.get_args_parser()
    args = parser.parse_args([])  # 单卡：不读命令行

    # ===== 对齐 r50_train_multi.sh =====
    args.dataset_file = "vid_multi"
    args.backbone = "resnet50"
    args.epochs = 7
    args.num_feature_levels = 1
    args.num_queries = 300
    args.dilation = True
    args.batch_size = 1
    args.num_ref_frames = 14
    args.lr_drop_epochs = [4, 6]
    args.num_workers = 16
    args.with_box_refine = True
    args.lr = 2e-4

    # ===== 你的数据/类别 =====
    args.vid_path = VID_PATH
    args.num_classes = 3

    # ===== 从单帧初始化（多帧阶段必须这样）=====
    args.coco_pretrain = False
    args.resume = SINGLE_CKPT

    # ===== 输出 =====
    args.output_dir = "exps/multibaseline/mydata_r50_multi_nf1_nr14_dc5"
    os.makedirs(args.output_dir, exist_ok=True)

    # ===== 设备 =====
    args.device = "cuda"

    train_main.main(args)

if __name__ == "__main__":
    run_multi()
