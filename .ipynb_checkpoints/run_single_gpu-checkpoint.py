import os
import main as train_main

# ====== 只需要改这两处 ======
VID_PATH = "/root/autodl-tmp/diyiqi_honwai_augment_1C"  # 你的数据集根目录
COCO_PRETRAIN = "/root/autodl-tmp/r50_deformable_detr_single_scale_dc5-checkpoint.pth"  # 你的预训练权重
# ============================

def run_singlebaseline():
    parser = train_main.get_args_parser()
    args = parser.parse_args([])  # 不读命令行

    # ====== 关键：对齐 configs/r50_train_single.sh ======
    args.dataset_file = "vid_single"
    args.vid_path = VID_PATH

    args.epochs = 50
    args.batch_size = 4           # 单卡如果显存不够，改成 2 或 1
    args.num_workers = 8          # 容器/IO慢可改 2 或 0

    args.num_feature_levels = 1   # ★必须：匹配 single_scale 权重
    args.num_queries = 300
    args.dilation = True          # ★必须：匹配 dc5
    args.with_box_refine = True
    args.lr_drop_epochs = [6, 7]  # 对齐作者脚本
    args.lr = 2e-4

    # 类别数（你的数据：car/van/truck）
    args.num_classes = 3

    # 预训练加载方式（对齐作者脚本里的 --coco_pretrain）
    args.coco_pretrain = True
    args.resume = COCO_PRETRAIN

    # 输出目录（你也可以改成自己喜欢的名字）
    args.output_dir = "exps/singlebaseline/mydata_r50_e8_singleScale_dc5"
    os.makedirs(args.output_dir, exist_ok=True)

    # 设备
    args.device = "cuda"
    # ================================================

    train_main.main(args)

if __name__ == "__main__":
    run_singlebaseline()
