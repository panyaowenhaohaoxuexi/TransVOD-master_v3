#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
三模态数据集完整验证脚本（方案A：vis/ir/sar 三个 root，内部 Data/... 相对路径一致）

验证内容：
1) 读取 COCO/VID 风格 json（images/annotations），遍历 images[*].file_name
2) 检查 VIS/IR/SAR 三模态文件是否存在
3) 检查三模态文件是否可打开
4) 检查三模态尺寸是否一致
5) （可选）检查多帧参考帧可用性：对每个样本在同一 video 目录下按偏移取参考帧，验证三模态存在/可打开/尺寸一致
   - 该参考帧检查是“通用版本”（按文件名数字序列推邻帧）。如果你的 vid_multi 采样策略不同，可按同样接口替换。

使用示例：
python check_triple_modal_dataset.py \
  --vis_root /path/to/vis_root \
  --ir_root  /path/to/ir_root \
  --sar_root /path/to/sar_root \
  --ann_json /path/to/vis_root/annotations/imagenet_vid_train.json \
  --data_dir Data \
  --max_check 0 \
  --check_refs \
  --ref_offsets -1 -2 -3

说明：
- --max_check 0 表示全量检查
- --check_refs 开启参考帧检查（仅在你准备做多帧训练时建议开启）
- --ref_offsets 指定参考帧相对当前帧的偏移（例如 -1 -2 -3 表示取前三帧；也可加 +1 +2）
"""

import os
import json
import argparse
from collections import Counter, defaultdict
from typing import List, Tuple, Optional

from PIL import Image


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_open_image(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        # 只验证能打开与尺寸，模式不强制；如果你想统一 RGB，可在这里 convert("RGB")
        return img
    except Exception:
        return None


def join_path(root: str, data_dir: str, file_name: str) -> str:
    # file_name 期望形如 train/video_0001/000123.jpg 或 Data/train/video_0001/000123.jpg
    # 兼容两种写法：如果 file_name 已含 Data/ 前缀，则不重复拼 data_dir
    norm = file_name.replace("\\", "/")
    if norm.startswith(data_dir.rstrip("/") + "/"):
        rel = norm
    else:
        rel = os.path.join(data_dir, norm).replace("\\", "/")
    return os.path.join(root, rel)


def parse_frame_index_from_filename(file_name: str) -> Optional[int]:
    """
    尝试从文件名中解析帧序号（假设 basename 形如 000123.jpg 或 123.png）
    返回 int 或 None（无法解析则 None）
    """
    base = os.path.basename(file_name)
    stem, _ext = os.path.splitext(base)
    if stem.isdigit():
        return int(stem)
    return None


def replace_frame_index_in_filename(file_name: str, new_idx: int) -> Optional[str]:
    """
    把 file_name 的 basename 数字部分替换为 new_idx，保持位数（例如 000123 -> 000120）
    如果原 basename 不是纯数字，返回 None
    """
    base = os.path.basename(file_name)
    stem, ext = os.path.splitext(base)
    if not stem.isdigit():
        return None
    width = len(stem)
    new_stem = str(new_idx).zfill(width)
    return os.path.join(os.path.dirname(file_name), new_stem + ext).replace("\\", "/")


def check_one_triplet(vis_path: str, ir_path: str, sar_path: str) -> Tuple[bool, str]:
    """
    返回 (ok, error_type)
    error_type: missing_vis, missing_ir, missing_sar, open_fail, size_mismatch
    """
    if not os.path.exists(vis_path):
        return False, "missing_vis"
    if not os.path.exists(ir_path):
        return False, "missing_ir"
    if not os.path.exists(sar_path):
        return False, "missing_sar"

    vis = safe_open_image(vis_path)
    ir = safe_open_image(ir_path)
    sar = safe_open_image(sar_path)
    if vis is None or ir is None or sar is None:
        return False, "open_fail"

    if vis.size != ir.size or vis.size != sar.size:
        return False, "size_mismatch"

    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis_root", required=True, type=str, help="VIS 根目录（包含 annotations/ 和 Data/）")
    ap.add_argument("--ir_root", required=True, type=str, help="IR 根目录（包含 Data/）")
    ap.add_argument("--sar_root", required=True, type=str, help="SAR 根目录（包含 Data/）")
    ap.add_argument("--ann_json", required=True, type=str, help="VIS annotations 下的 json 路径")
    ap.add_argument("--data_dir", default="Data", type=str, help="图像数据子目录名，默认 Data")
    ap.add_argument("--max_check", default=0, type=int, help="最多检查多少张图；0 表示全量")
    ap.add_argument("--show_examples", default=10, type=int, help="每类错误最多打印多少条示例")

    # reference frames checking
    ap.add_argument("--check_refs", action="store_true", help="开启参考帧检查（多帧训练前建议开启）")
    ap.add_argument(
        "--ref_offsets",
        nargs="*",
        type=int,
        default=[-1, -2, -3],
        help="参考帧相对当前帧的偏移列表，例如 -1 -2 -3 或 -1 1",
    )
    args = ap.parse_args()

    vis_root = args.vis_root
    ir_root = args.ir_root
    sar_root = args.sar_root
    ann_json = args.ann_json
    data_dir = args.data_dir

    data = load_json(ann_json)
    images = data.get("images", [])
    print("images in json:", len(images))

    errors = Counter()
    examples = defaultdict(list)

    checked = 0
    checked_refs = 0
    ref_errors = Counter()
    ref_examples = defaultdict(list)

    for img in images:
        file_name = img.get("file_name")
        if not file_name:
            errors["missing_file_name_field"] += 1
            continue

        vis_path = join_path(vis_root, data_dir, file_name)
        ir_path = join_path(ir_root, data_dir, file_name)
        sar_path = join_path(sar_root, data_dir, file_name)

        ok, et = check_one_triplet(vis_path, ir_path, sar_path)
        if not ok:
            errors[et] += 1
            if len(examples[et]) < args.show_examples:
                examples[et].append(file_name)
            continue

        checked += 1

        # 参考帧检查：根据 basename 数字推邻帧（通用方案）
        if args.check_refs:
            cur_idx = parse_frame_index_from_filename(file_name)
            if cur_idx is None:
                ref_errors["ref_parse_fail"] += 1
                if len(ref_examples["ref_parse_fail"]) < args.show_examples:
                    ref_examples["ref_parse_fail"].append(file_name)
            else:
                for off in args.ref_offsets:
                    ref_name = replace_frame_index_in_filename(file_name, cur_idx + off)
                    if ref_name is None:
                        ref_errors["ref_replace_fail"] += 1
                        if len(ref_examples["ref_replace_fail"]) < args.show_examples:
                            ref_examples["ref_replace_fail"].append(file_name)
                        continue

                    vref = join_path(vis_root, data_dir, ref_name)
                    iref = join_path(ir_root, data_dir, ref_name)
                    sref = join_path(sar_root, data_dir, ref_name)

                    rok, ret = check_one_triplet(vref, iref, sref)
                    if not rok:
                        ref_errors[ret] += 1
                        if len(ref_examples[ret]) < args.show_examples:
                            ref_examples[ret].append(ref_name)
                    else:
                        checked_refs += 1

        if args.max_check and checked >= args.max_check:
            break

    print("\n=== Single-frame triplet check ===")
    print("checked ok:", checked)
    print("errors:", dict(errors))
    if errors:
        print("\nExamples (up to %d each):" % args.show_examples)
        for k, lst in examples.items():
            print(f"  {k}:")
            for x in lst:
                print("    ", x)

    if args.check_refs:
        print("\n=== Reference-frame check (generic by filename index) ===")
        print("ref_offsets:", args.ref_offsets)
        print("checked refs ok:", checked_refs)
        print("ref errors:", dict(ref_errors))
        if ref_errors:
            print("\nRef examples (up to %d each):" % args.show_examples)
            for k, lst in ref_examples.items():
                print(f"  {k}:")
                for x in lst:
                    print("    ", x)

    # 最终判定
    ok_single = (sum(errors.values()) == 0)
    ok_refs = (not args.check_refs) or (sum(ref_errors.values()) == 0)
    print("\n=== RESULT ===")
    if ok_single and ok_refs:
        print("PASS: dataset triplets (and refs if enabled) look consistent.")
    else:
        print("FAIL: please fix above error categories.")


if __name__ == "__main__":
    main()




# python 三模态数据集-验证是否整理正确v2.py \
#   --vis_root /mnt/f/Tri_modal_data/Temp_Tri_modal_data/vis_root \
#   --ir_root /mnt/f/Tri_modal_data/Temp_Tri_modal_data/ir_root \
#   --sar_root /mnt/f/Tri_modal_data/Temp_Tri_modal_data/sar_root \
#   --ann_json /mnt/f/Tri_modal_data/Temp_Tri_modal_data/ir_root/annotations/imagenet_vid_train.json \
#   --data_dir Data



# "F:\Tri_modal_data\Temp_Tri_modal_data\ir_root\annotations\imagenet_vid_train.json"