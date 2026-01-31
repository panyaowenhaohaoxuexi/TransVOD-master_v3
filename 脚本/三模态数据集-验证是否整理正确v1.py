import os
import json
from PIL import Image
from collections import Counter

VIS_ROOT = "/mnt/f/Tri_modal_data/Temp_Tri_modal_data/vis_root"
IR_ROOT  = "/mnt/f/Tri_modal_data/Temp_Tri_modal_data/ir_root"
SAR_ROOT = "/mnt/f/Tri_modal_data/Temp_Tri_modal_data/sar_root"
ANN_PATH = os.path.join(IR_ROOT, "annotations", "imagenet_vid_train.json")  # 或 val.json
# /mnt/f/Tri_modal_data/Temp_Tri_modal_data/labels/val
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def main(max_check=None):
    data = load_json(ANN_PATH)
    images = data.get("images", [])
    print("images in json:", len(images))

    errors = Counter()
    checked = 0

    for img in images:
        file_name = img["file_name"]   # 期望类似 "train/video_0001/000123.jpg"
        vis_path = os.path.join(VIS_ROOT, "Data", file_name)
        ir_path  = os.path.join(IR_ROOT,  "Data", file_name)
        sar_path = os.path.join(SAR_ROOT, "Data", file_name)

        # 1) existence
        if not os.path.exists(vis_path):
            errors["missing_vis"] += 1
            continue
        if not os.path.exists(ir_path):
            errors["missing_ir"] += 1
            continue
        if not os.path.exists(sar_path):
            errors["missing_sar"] += 1
            continue

        # 2) open + size
        try:
            vis = Image.open(vis_path)
            ir  = Image.open(ir_path)
            sar = Image.open(sar_path)
        except Exception:
            errors["open_fail"] += 1
            continue

        if vis.size != ir.size or vis.size != sar.size:
            errors["size_mismatch"] += 1
            continue

        checked += 1
        if max_check is not None and checked >= max_check:
            break

    print("checked ok:", checked)
    print("errors:", dict(errors))

if __name__ == "__main__":
    main(max_check=None)  # 可改成 5000 先快速跑
