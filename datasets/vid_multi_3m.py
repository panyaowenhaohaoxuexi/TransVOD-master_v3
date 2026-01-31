# Modified by Lu He
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from pathlib import Path
import os
from io import BytesIO
import random

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .coco_video_parser import CocoVID
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_multi as T
from torch.utils.data.dataset import ConcatDataset


class CocoDetection3M(TvCocoDetection):
    """
    Tri-modal multi-frame dataset (Scheme A):
      - VIS root: args.vid_path (has annotations/ and Data/)
      - IR  root: args.vid_path_ir (has Data/)
      - SAR root: args.vid_path_sar (has Data/)
    The json file_name is used as-is relative to the img_folder root passed into TvCocoDetection.
    """

    def __init__(self, img_folder_vis, img_folder_ir, img_folder_sar,
                 ann_file, transforms, return_masks,
                 interval1, interval2, num_ref_frames=3,
                 is_train=True, filter_key_img=True,
                 cache_mode=False, local_rank=0, local_size=1):

        super(CocoDetection3M, self).__init__(
            img_folder_vis, ann_file,
            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size
        )
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

        self.ann_file = ann_file
        self.frame_range = [-2, 2]
        self.num_ref_frames = num_ref_frames
        self.cocovid = CocoVID(self.ann_file)

        self.is_train = is_train
        self.filter_key_img = filter_key_img
        self.interval1 = interval1
        self.interval2 = interval2

        # roots for IR/SAR (TvCocoDetection only knows VIS root)
        self.root_ir = str(img_folder_ir)
        self.root_sar = str(img_folder_sar)

        self.cache_mode = cache_mode
        self.cache_ir = {} if cache_mode else None
        self.cache_sar = {} if cache_mode else None

    def _open_from_root(self, root_dir: str, rel_path: str, cache_dict=None):
        full_path = os.path.join(root_dir, rel_path)
        if self.cache_mode:
            if rel_path not in cache_dict:
                with open(full_path, "rb") as f:
                    cache_dict[rel_path] = f.read()
            return Image.open(BytesIO(cache_dict[rel_path])).convert("RGB")
        from PIL import Image
        return Image.open(full_path).convert("RGB")

    def _get_triplet(self, file_name: str):
        # VIS uses parent's get_image (supports its cache_mode)
        img_vis = self.get_image(file_name)
        img_ir = self._open_from_root(self.root_ir, file_name, self.cache_ir)
        img_sar = self._open_from_root(self.root_sar, file_name, self.cache_sar)

        if img_ir.size != img_vis.size or img_sar.size != img_vis.size:
            raise ValueError(
                f"3-modal size mismatch for {file_name}: "
                f"vis={img_vis.size}, ir={img_ir.size}, sar={img_sar.size}"
            )
        return img_vis, img_ir, img_sar

    def __getitem__(self, idx):
        imgs = []
        coco = self.coco

        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        path = img_info["file_name"]
        video_id = img_info["video_id"]

        # current frame triplet
        img_vis, img_ir, img_sar = self._get_triplet(path)

        target = {"image_id": img_id, "annotations": anns}
        img_vis, target = self.prepare(img_vis, target)

        # IMPORTANT: append in fixed order
        imgs.extend([img_vis, img_ir, img_sar])

        if video_id == -1:  # imgnet_det
            for _ in range(self.num_ref_frames):
                imgs.extend([img_vis, img_ir, img_sar])
        else:  # imgnet_vid
            img_ids = self.cocovid.get_img_ids_from_vid(video_id)
            ref_img_ids = []

            if self.is_train:  # Train
                interval = self.num_ref_frames + 2
                left = max(img_ids[0], img_id - interval)
                right = min(img_ids[-1], img_id + interval)
                sample_range = list(range(left, right + 1))

                if self.num_ref_frames >= 10:
                    sample_range = img_ids

                if self.filter_key_img and img_id in sample_range:
                    sample_range.remove(img_id)

                while len(sample_range) < self.num_ref_frames:
                    sample_range.extend(sample_range)

                ref_img_ids = random.sample(sample_range, self.num_ref_frames)

            else:  # Eval
                Len = len(img_ids)
                interval = max(int(Len // 16), 1)

                if self.num_ref_frames < 8:
                    left_indexs = int((img_id - img_ids[0]) // interval)
                    if left_indexs < self.num_ref_frames:
                        for i in range(self.num_ref_frames):
                            ref_img_ids.append(min(img_id + (i + 1) * interval, img_ids[-1]))
                    else:
                        for i in range(self.num_ref_frames):
                            ref_img_ids.append(max(img_id - (i + 1) * interval, img_ids[0]))
                else:
                    sample_range = []
                    left_indexs = int((img_ids[0] - img_id) // interval)
                    right_indexs = int((img_ids[-1] - img_id) // interval)
                    for i in range(left_indexs, right_indexs):
                        if i < 0:
                            index = max(img_id + i * interval, img_ids[0])
                            sample_range.append(index)
                        elif i > 0:
                            index = min(img_id + i * interval, img_ids[-1])
                            sample_range.append(index)

                    if self.filter_key_img and img_id in sample_range:
                        sample_range.remove(img_id)

                    while len(sample_range) < self.num_ref_frames:
                        sample_range.extend(sample_range)

                    ref_img_ids = sample_range[:self.num_ref_frames]

            for ref_img_id in ref_img_ids:
                ref_img_info = coco.loadImgs(ref_img_id)[0]
                ref_path = ref_img_info["file_name"]

                ref_vis, ref_ir, ref_sar = self._get_triplet(ref_path)
                imgs.extend([ref_vis, ref_ir, ref_sar])

        if self._transforms is not None:
            imgs, target = self._transforms(imgs, target)

        return torch.cat(imgs, dim=0), target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set in ("train_vid", "train_det", "train_joint"):
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    if image_set == "val":
        return T.Compose([
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    root_vis = Path(args.vid_path)
    root_ir = Path(args.vid_path_ir) if getattr(args, "vid_path_ir", "") else root_vis
    root_sar = Path(args.vid_path_sar) if getattr(args, "vid_path_sar", "") else root_vis

    assert root_vis.exists(), f"provided VIS path {root_vis} does not exist"
    assert root_ir.exists(), f"provided IR path {root_ir} does not exist"
    assert root_sar.exists(), f"provided SAR path {root_sar} does not exist"

    PATHS = {
        "train_det": [(root_vis / "Data" / "DET", root_vis / "annotations" / "imagenet_det_30plus1cls_vid_train.json")],
        "train_vid": [(root_vis / "Data",         root_vis / "annotations" / "imagenet_vid_train.json")],
        "train_joint":[(root_vis / "Data",        root_vis / "annotations" / "imagenet_vid_train_joint_30.json")],
        "val":       [(root_vis / "Data",         root_vis / "annotations" / "imagenet_vid_val.json")],
    }

    datasets = []
    for (img_folder_vis, ann_file) in PATHS[image_set]:
        # make ir/sar img folders mirror vis folder (Data or Data/DET)
        rel = img_folder_vis.relative_to(root_vis)
        img_folder_ir = root_ir / rel
        img_folder_sar = root_sar / rel

        dataset = CocoDetection3M(
            img_folder_vis=img_folder_vis,
            img_folder_ir=img_folder_ir,
            img_folder_sar=img_folder_sar,
            ann_file=ann_file,
            transforms=make_coco_transforms(image_set),
            is_train=(not args.eval),
            interval1=args.interval1,
            interval2=args.interval2,
            num_ref_frames=args.num_ref_frames,
            return_masks=args.masks,
            cache_mode=args.cache_mode,
            filter_key_img=True,
            local_rank=get_local_rank(),
            local_size=get_local_size(),
        )
        datasets.append(dataset)

    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
