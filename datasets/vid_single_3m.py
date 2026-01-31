from pathlib import Path
import os
from io import BytesIO
from PIL import Image

import torch
from torch.utils.data.dataset import ConcatDataset

from .torchvision_datasets import CocoDetection as TvCocoDetection
from .vid_single import ConvertCocoPolysToMask
import datasets.transforms_multi as T  # 用 multi transforms 保证多图同步增强
from util.misc import get_local_rank, get_local_size


class CocoDetection3M(TvCocoDetection):
    """
    Scheme A:
      vis_root: has annotations/ and Data/
      ir_root : has Data/ (same relative paths as vis)
      sar_root: has Data/ (same relative paths as vis)
    JSON file_name is used as-is (relative to Data/ or including Data/ depending on your json).
    """

    def __init__(self, img_folder_vis, img_folder_ir, img_folder_sar,
                 ann_file, transforms, return_masks,
                 cache_mode=False, local_rank=0, local_size=1):

        # NOTE: TvCocoDetection uses img_folder as "root" to join with file_name
        super().__init__(img_folder_vis, ann_file,
                         cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)

        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

        self.root_vis = str(img_folder_vis)
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
        return Image.open(full_path).convert("RGB")

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        file_name = coco.loadImgs(img_id)[0]["file_name"]

        # VIS: use parent's get_image (supports its cache_mode)
        img_vis = self.get_image(file_name)

        # IR/SAR: open from their roots
        img_ir = self._open_from_root(self.root_ir, file_name, self.cache_ir)
        img_sar = self._open_from_root(self.root_sar, file_name, self.cache_sar)

        # enforce same size (registered data should satisfy)
        if img_ir.size != img_vis.size or img_sar.size != img_vis.size:
            raise ValueError(
                f"3-modal size mismatch for {file_name}: "
                f"vis={img_vis.size}, ir={img_ir.size}, sar={img_sar.size}"
            )

        target = {"image_id": img_id, "annotations": anns}

        # target computed based on VIS image size; registration assumption makes it valid for all
        img_vis, target = self.prepare(img_vis, target)

        imgs = [img_vis, img_ir, img_sar]  # list[PIL]
        if self._transforms is not None:
            imgs, target = self._transforms(imgs, target)  # list[Tensor(3,H,W)]

        # concat channels -> Tensor(9,H,W)
        return torch.cat(imgs, dim=0), target


def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

    raise ValueError(f"unknown image_set: {image_set}")


def build(image_set, args):
    root_vis = Path(args.vid_path)
    root_ir = Path(args.vid_path_ir) if getattr(args, "vid_path_ir", "") else root_vis
    root_sar = Path(args.vid_path_sar) if getattr(args, "vid_path_sar", "") else root_vis

    assert root_vis.exists(), f"VIS root not exists: {root_vis}"
    assert root_ir.exists(), f"IR root not exists: {root_ir}"
    assert root_sar.exists(), f"SAR root not exists: {root_sar}"

    PATHS = {
        "train_det": [(root_vis / "Data" / "DET", root_vis / "annotations" / "imagenet_det_30plus1cls_vid_train.json")],
        "train_vid": [(root_vis / "Data", root_vis / "annotations" / "imagenet_vid_train.json")],
        "train_joint": [(root_vis / "Data", root_vis / "annotations" / "imagenet_vid_train_joint_30.json")],
        "val": [(root_vis / "Data", root_vis / "annotations" / "imagenet_vid_val.json")],
    }

    datasets = []
    for (img_folder_vis, ann_file) in PATHS[image_set]:
        img_folder_ir = root_ir / "Data"
        img_folder_sar = root_sar / "Data"

        ds = CocoDetection3M(
            img_folder_vis=img_folder_vis,
            img_folder_ir=img_folder_ir,
            img_folder_sar=img_folder_sar,
            ann_file=ann_file,
            transforms=make_coco_transforms(image_set),
            return_masks=args.masks,
            cache_mode=args.cache_mode,
            local_rank=get_local_rank(),
            local_size=get_local_size(),
        )
        datasets.append(ds)

    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
