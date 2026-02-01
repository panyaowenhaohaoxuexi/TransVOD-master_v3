import main as train_main
from datasets import build_dataset
import torch

VIS_PATH = "/mnt/f/Tri_modal_data/Temp_Tri_modal_data/vis_root"
IR_PATH  = "/mnt/f/Tri_modal_data/Temp_Tri_modal_data/ir_root"
SAR_PATH = "/mnt/f/Tri_modal_data/Temp_Tri_modal_data/sar_root"

parser = train_main.get_args_parser()
args = parser.parse_args([])

args.dataset_file = "vid_multi_3m"
args.vid_path = VIS_PATH
args.vid_path_ir = IR_PATH
args.vid_path_sar = SAR_PATH
args.num_classes = 3
args.num_ref_frames = 14
args.eval = True
args.masks = False
args.cache_mode = False
args.interval1 = 1
args.interval2 = 1

ds = build_dataset("val", args)
x, tgt = ds[0]

print("x.shape:", x.shape)  # 期望 C = 9*(1+K)
print("labels unique:", torch.unique(tgt["labels"]).tolist())
if tgt["labels"].numel() > 0:
    print("labels max:", tgt["labels"].max().item())
