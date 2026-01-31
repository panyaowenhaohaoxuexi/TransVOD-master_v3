import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# 引入项目依赖
import main as train_main
from models import build_model

# ==========================================
# 1. 配置参数
# ==========================================
def get_args():
    parser = train_main.get_args_parser()
    args = parser.parse_args([])
    
    # 关键修复：指定为单帧模式
    args.dataset_file = 'vid_single' 

    args.num_classes = 3           # 你的类别数
    args.num_feature_levels = 1    # 单尺度
    args.dilation = True           # DC5
    args.with_box_refine = True    
    
    args.device = 'cuda'
    args.lr_backbone = 0           
    args.masks = False             
    
    return args

# ==========================================
# 2. 图像预处理
# ==========================================
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. 核心推理函数
# ==========================================
def detect(img_path, output_path, checkpoint_path, threshold=0.5):
    args = get_args()
    print(f"Loading model from {checkpoint_path}...")
    
    model, _, _ = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()

    # 读取图像
    im_pil = Image.open(img_path).convert('RGB')
    w, h = im_pil.size
    img_tensor = transform(im_pil).unsqueeze(0).to(args.device)
    im_cv = cv2.imread(img_path)

    # 推理
    print("Inference...")
    with torch.no_grad():
        outputs = model(img_tensor)

    # --- D. 解析结果 (修复版) ---
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    
    # 拿到最大分数和类别
    max_scores, max_labels = probas.max(-1)
    
    # 筛选
    keep = max_scores > threshold
    
    scores = max_scores[keep]
    labels = max_labels[keep]
    bboxes_scaled = outputs['pred_boxes'][0, keep]
    
    # 坐标还原
    bboxes_scaled[:, 0] = bboxes_scaled[:, 0] * w
    bboxes_scaled[:, 1] = bboxes_scaled[:, 1] * h
    bboxes_scaled[:, 2] = bboxes_scaled[:, 2] * w
    bboxes_scaled[:, 3] = bboxes_scaled[:, 3] * h
    
    boxes = []
    for bbox in bboxes_scaled:
        cx, cy, bw, bh = bbox.tolist()
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)
        boxes.append([x1, y1, x2, y2])

    print(f"Detected {len(boxes)} objects with threshold > {threshold}")

    # --- E. 画图 ---
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)] 
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cls_id = labels[i].item()
        score = scores[i].item()  # 现在这里不会报错了
        
        color = colors[cls_id % len(colors)]
        cv2.rectangle(im_cv, (x1, y1), (x2, y2), color, 2)
        text = f"Class {cls_id}: {score:.2f}"
        cv2.putText(im_cv, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(output_path, im_cv)
    print(f"Result saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--out', type=str, default='result.jpg')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--thresh', type=float, default=0.7)
    args = parser.parse_args()
    
    detect(args.img, args.out, args.ckpt, args.thresh)
    
    
    # python single_predict.py --img /root/autodl-tmp/Tri_modal_data/Data/train/video_0001/0139.jpg --ckpt /root/TransVOD-master/exps/singlebaseline/mydata_r50_e8_singleScale_dc5/checkpoint0049.pth --out /root/TransVOD-master/result_0139.jpg --thresh 0.3