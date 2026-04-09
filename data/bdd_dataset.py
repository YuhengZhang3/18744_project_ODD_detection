import os
import json
from glob import glob
from collections import Counter

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# class order must match configs/odd_config.py
TIME_CLASSES = ["dawn/dusk", "daytime", "night", "undefined"]
SCENE_CLASSES = [
    "city street",
    "gas stations",
    "highway",
    "parking lot",
    "residential",
    "tunnel",
    "undefined",
]
VIS_CLASSES = ["poor", "medium", "good"]
DRIVABLE_CLASSES = ["background", "alternative_drivable", "direct_drivable"]

TIME_MAP = {n: i for i, n in enumerate(TIME_CLASSES)}
SCENE_MAP = {n: i for i, n in enumerate(SCENE_CLASSES)}
VIS_MAP = {n: i for i, n in enumerate(VIS_CLASSES)}
DRIVABLE_MAP = {n: i for i, n in enumerate(DRIVABLE_CLASSES)}


def get_default_transform():
    return transforms.Compose(
        [
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_default_mask_resize():
    return transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.NEAREST)


def get_bdd_root():
    candidates = [
        "/home/yuhengz3@andrew.cmu.edu/bdd100k",
        "/Users/zhangyuheng/Documents/Study/CMU_Courses/2026Spring/18744/Project/datasets",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise RuntimeError("bdd root not found, update get_bdd_root()")


def resolve_img_root_for_split(root, split):
    cand1 = os.path.join(root, "100k_datasets", split)
    cand2 = os.path.join(root, "100k_datasets", "100k", split)
    for c in (cand1, cand2):
        if os.path.isdir(c):
            return c
    raise RuntimeError(f"no img dir for split {split} under {root}")


def resolve_label_root_for_split(root, split):
    cand1 = os.path.join(root, "100k_label", split)
    cand2 = os.path.join(root, "100k_label", "100k", split)
    for c in (cand1, cand2):
        if os.path.isdir(c):
            return c
    raise RuntimeError(f"no label dir for split {split} under {root}")


def resolve_visibility_root_for_split(root, split):
    c = os.path.join(root, "visibility_labels", split)
    if os.path.isdir(c):
        return c
    raise RuntimeError(f"no visibility label dir for split {split} under {root}")


def resolve_drivable_root_for_split(root, split):
    c = os.path.join(root, "bdd100k_drivable_maps", "labels", split)
    if os.path.isdir(c):
        return c
    raise RuntimeError(f"no drivable label dir for split {split} under {root}")


class BDDDTimeScene(Dataset):
    # bdd100k time + scene from per-image json files
    # keep init light: do not preload all labels

    def __init__(self, img_root, label_dir, transform=None):
        self.img_root = img_root
        self.label_paths = sorted(glob(os.path.join(label_dir, "*.json")))
        self.transform = transform if transform is not None else get_default_transform()

        self.targets_time = None
        self.targets_scene = None

    def __len__(self):
        return len(self.label_paths)

    def _load_attrs(self, label_path):
        with open(label_path, "r") as f:
            data = json.load(f)
        attrs = data.get("attributes", {})
        t_str = attrs.get("timeofday", "undefined")
        s_str = attrs.get("scene", "undefined")
        return t_str, s_str

    def __getitem__(self, idx):
        label_path = self.label_paths[idx]
        base = os.path.splitext(os.path.basename(label_path))[0]
        img_path = os.path.join(self.img_root, base + ".jpg")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        t_str, s_str = self._load_attrs(label_path)
        t_label = TIME_MAP.get(t_str, TIME_MAP["undefined"])
        s_label = SCENE_MAP.get(s_str, SCENE_MAP["undefined"])

        labels = {
            "time": torch.tensor(t_label, dtype=torch.long),
            "scene": torch.tensor(s_label, dtype=torch.long),
        }

        mask = {
            "time": torch.tensor(1.0, dtype=torch.float32),
            "scene": torch.tensor(1.0, dtype=torch.float32),
        }

        severity = {
            "time": torch.tensor(1.0, dtype=torch.float32),
            "scene": torch.tensor(1.0, dtype=torch.float32),
        }

        return {
            "image": img,
            "labels": labels,
            "mask": mask,
            "severity": severity,
        }


def collate_time_scene(batch):
    imgs = torch.stack([b["image"] for b in batch], dim=0)

    out_labels = {}
    out_mask = {}
    out_sev = {}

    for key in ["time", "scene"]:
        out_labels[key] = torch.stack([b["labels"][key] for b in batch], dim=0)
        out_mask[key] = torch.stack([b["mask"][key] for b in batch], dim=0)
        out_sev[key] = torch.stack([b["severity"][key] for b in batch], dim=0)

    return {
        "images": imgs,
        "labels": out_labels,
        "mask": out_mask,
        "severity": out_sev,
    }


class BDDDVisibility(Dataset):
    # visibility labels from *_vis.json
    # keep init light: do not preload all labels

    def __init__(self, split, transform=None):
        self.bdd_root = get_bdd_root()
        self.img_root = resolve_img_root_for_split(self.bdd_root, split)
        self.vis_root = resolve_visibility_root_for_split(self.bdd_root, split)
        self.transform = transform if transform is not None else get_default_transform()

        self.label_paths = sorted(glob(os.path.join(self.vis_root, "*_vis.json")))
        self.targets = None

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, idx):
        label_path = self.label_paths[idx]
        base = os.path.basename(label_path).replace("_vis.json", "")
        img_path = os.path.join(self.img_root, base + ".jpg")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        with open(label_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if "visibility" in data:
                y = data["visibility"]
            elif "label" in data:
                y = data["label"]
            else:
                raise ValueError(f"unknown visibility label format: {label_path}")
        else:
            y = data

        label = torch.tensor(int(y), dtype=torch.long)
        return img, label



class BDDDDrivable(Dataset):
    # drivable segmentation labels from bdd100k_drivable_maps/labels/{split}
    # returns image + segmentation mask for drivable head
    # keep init light: do not scan all masks here

    def __init__(self, split, transform=None, mask_resize=True):
        assert split in {"train", "val"}
        self.split = split
        self.bdd_root = get_bdd_root()
        self.img_root = resolve_img_root_for_split(self.bdd_root, split)
        self.mask_root = resolve_drivable_root_for_split(self.bdd_root, split)

        self.transform = transform if transform is not None else get_default_transform()
        self.mask_resize = get_default_mask_resize() if mask_resize else None

        self.mask_paths = sorted(glob(os.path.join(self.mask_root, "*_drivable_id.png")))
        self.targets = [0 for _ in self.mask_paths]  # placeholder only

    def __len__(self):
        return len(self.mask_paths)

    def get_pixel_counts(self):
        # use cached stats instead of scanning masks every time
        from configs.data_stats import BDD_DRIVABLE_PIXEL_COUNTS
        return Counter(BDD_DRIVABLE_PIXEL_COUNTS)

    def get_mean_pixel_ratios(self):
        counts = self.get_pixel_counts()
        total = sum(counts.values())
        return {
            "background": counts.get(0, 0) / total if total > 0 else 0.0,
            "alternative_drivable": counts.get(1, 0) / total if total > 0 else 0.0,
            "direct_drivable": counts.get(2, 0) / total if total > 0 else 0.0,
        }

    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        base = os.path.basename(mask_path).replace("_drivable_id.png", "")
        img_path = os.path.join(self.img_root, base + ".jpg")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        mask = Image.open(mask_path).convert("L")
        if self.mask_resize is not None:
            mask = self.mask_resize(mask)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        labels = {
            "drivable": mask,
        }
        batch_mask = {
            "drivable": torch.tensor(1.0, dtype=torch.float32),
        }
        severity = {
            "drivable": torch.tensor(1.0, dtype=torch.float32),
        }

        return {
            "image": img,
            "labels": labels,
            "mask": batch_mask,
            "severity": severity,
        }

def collate_drivable(batch):
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["labels"]["drivable"] for b in batch], dim=0)
    batch_mask = torch.stack([b["mask"]["drivable"] for b in batch], dim=0)
    severity = torch.stack([b["severity"]["drivable"] for b in batch], dim=0)

    return {
        "images": imgs,
        "labels": {
            "drivable": masks,
        },
        "mask": {
            "drivable": batch_mask,
        },
        "severity": {
            "drivable": severity,
        },
    }
