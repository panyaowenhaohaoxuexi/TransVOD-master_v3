# txt_to_imagenet_vid_json.py
# 作用：把 YOLO txt（cls cx cy w h，归一化）按 video_000x 子目录结构转换成 TransVOD/VID 所需的 JSON
# 输出：imagenet_vid_train.json 或 imagenet_vid_val.json

import os
import json
import glob
from PIL import Image

# ================== 你只需要改这里 ==================
# 数据集根目录
VID_ROOT = r"/mnt/f/Tri_modal_data/Temp_Tri_modal_data/ir_root"
# F:\Tri_modal_data\Temp_Tri_modal_data\ir_root
# INPUT_ROOT = r'/mnt/f/Tri_modal_data/Temp_Tri_modal_data/labels_val_txt'
# 1. 设置当前要生成的集： "train" 或 "val"
SPLIT = "val"  # 修改这里来切换训练集或验证集

# 2. 修改图像目录逻辑：去掉 "VID"
# 假设您的结构是：Data/train/video_0001/...
IMG_ROOT = os.path.join(VID_ROOT, "Data", SPLIT)

# 3. 修改标签目录逻辑：自动根据 SPLIT 选择对应的文件夹
# 当 SPLIT 是 "train" 时，自动找 labels_train_txt
# 当 SPLIT 是 "val" 时，自动找 labels_val_txt
LABEL_ROOT = os.path.join(VID_ROOT, f"labels_{SPLIT}_txt")

# 输出 JSON 路径
OUT_JSON = os.path.join(VID_ROOT, "annotations", f"imagenet_vid_{SPLIT}.json")

# 类别定义 (根据您的 txt 里的 id: 0, 1, 2 修改)
CATEGORIES = [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "van"},
    {"id": 3, "name": "truck"},
]
ALLOW_MISSING_TXT = True
IMG_EXTS = (".jpg", ".jpeg", ".png")
# ====================================================


def sorted_frame_files(folder: str):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    # 按文件名排序（保持时序），尽量按数字排序
    def key_fn(p):
        base = os.path.splitext(os.path.basename(p))[0]
        return (len(base), base)  # "0001" < "0100"
    return sorted(files, key=key_fn)


def yolo_line_to_xywh_pixels(line: str, W: int, H: int):
    # line: cls cx cy w h  (normalized)
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(parts[0])
    cx, cy, bw, bh = map(float, parts[1:5])

    x = (cx - bw / 2.0) * W
    y = (cy - bh / 2.0) * H
    ww = bw * W
    hh = bh * H

    # clip
    x = max(0.0, x)
    y = max(0.0, y)
    ww = max(0.0, min(ww, W - x))
    hh = max(0.0, min(hh, H - y))

    return cls, [x, y, ww, hh]


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    if not os.path.isdir(IMG_ROOT):
        raise RuntimeError(f"IMG_ROOT not found: {IMG_ROOT}")
    if not os.path.isdir(LABEL_ROOT):
        raise RuntimeError(f"LABEL_ROOT not found: {LABEL_ROOT}")

    # 识别有哪些 video_xxxx（以图像目录为准）
    video_names = sorted([
        d for d in os.listdir(IMG_ROOT)
        if os.path.isdir(os.path.join(IMG_ROOT, d))
    ])

    if not video_names:
        raise RuntimeError(f"No video folders found under IMG_ROOT: {IMG_ROOT}")

    data = {
        "videos": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

    video_id_map = {}  # name -> id
    image_id = 1
    ann_id = 1

    for v_idx, vname in enumerate(video_names, start=1):
        video_id_map[vname] = v_idx
        data["videos"].append({"id": v_idx, "name": vname})

        img_dir = os.path.join(IMG_ROOT, vname)
        label_dir = os.path.join(LABEL_ROOT, vname)

        img_files = sorted_frame_files(img_dir)
        if not img_files:
            print(f"[WARN] No images in {img_dir}, skip.")
            continue

        # 读取第一张图确定尺寸（要求同一视频内尺寸一致）
        with Image.open(img_files[0]) as im:
            W, H = im.size

        for frame_id, img_path in enumerate(img_files):
            base = os.path.splitext(os.path.basename(img_path))[0]

            # 尺寸检查（可选，但建议）
            with Image.open(img_path) as im:
                iw, ih = im.size
            if (iw, ih) != (W, H):
                raise RuntimeError(f"Image size mismatch in {vname}: {img_path} got {(iw, ih)} expected {(W, H)}")

            # JSON 中的 file_name 必须相对 Data/VID
            file_name = f"{SPLIT}/{vname}/{os.path.basename(img_path)}"

            cur_image_id = image_id
            data["images"].append({
                "id": cur_image_id,
                "file_name": file_name,
                "width": W,
                "height": H,
                "video_id": v_idx,
                "frame_id": frame_id
            })
            image_id += 1

            # 对应 txt
            txt_path = os.path.join(label_dir, base + ".txt")
            if not os.path.exists(txt_path):
                if not ALLOW_MISSING_TXT:
                    raise FileNotFoundError(f"Missing txt: {txt_path}")
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parsed = yolo_line_to_xywh_pixels(line, W, H)
                    if parsed is None:
                        continue

                    cls, bbox = parsed
                    if cls not in (0, 1, 2):
                        raise RuntimeError(f"Unexpected class id {cls} in {txt_path}")

                    data["annotations"].append({
                        "id": ann_id,
                        "image_id": cur_image_id,
                        "category_id": cls + 1,  # 0->1,1->2,2->3
                        "bbox": bbox,
                        "area": float(bbox[2] * bbox[3]),
                        "iscrowd": 0
                    })
                    ann_id += 1

        print(f"[OK] {vname}: images={len(img_files)} labels_dir_exists={os.path.isdir(label_dir)}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print("Wrote:", OUT_JSON)
    print("Videos:", len(data["videos"]))
    print("Images:", len(data["images"]))
    print("Annotations:", len(data["annotations"]))


if __name__ == "__main__":
    main()
