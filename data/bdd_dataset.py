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

    def __init__(self, img_root, label_dir, transform=None):
        self.img_root = img_root
        self.label_paths = sorted(glob(os.path.join(label_dir, "*.json")))
        self.transform = transform if transform is not None else get_default_transform()

        self.targets_time = []
        self.targets_scene = []

        for p in self.label_paths:
            t_str, s_str = self._load_attrs(p)
            self.targets_time.append(TIME_MAP.get(t_str, TIME_MAP["undefined"]))
            self.targets_scene.append(SCENE_MAP.get(s_str, SCENE_MAP["undefined"]))

    def __len__(self):
        return len(self.label_paths)

    def _load_attrs(self, label_path):
        with open(label_path, "r") as f:
            data = json.load(f)
        attrs = data.get("attributes", {})
        t_str = attrs.get("timeofday", "undefined")
        s_str = attrs.get("scene", "undefined")
        return t_str, s_str

    def get_class_counts(self):
        return {
            "time": Counter(self.targets_time),
            "scene": Counter(self.targets_scene),
        }

    def __getitem__(self, idx):
        label_path = self.label_paths[idx]
        base = os.path.splitext(os.path.basename(label_path))[0]
        img_path = os.path.join(self.img_root, base + ".jpg")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        t_label = self.targets_time[idx]
        s_label = self.targets_scene[idx]

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

    def __init__(self, split, transform=None):
        self.bdd_root = get_bdd_root()
        self.img_root = resolve_img_root_for_split(self.bdd_root, split)
        self.vis_root = resolve_visibility_root_for_split(self.bdd_root, split)
        self.transform = transform if transform is not None else get_default_transform()

        self.label_paths = sorted(glob(os.path.join(self.vis_root, "*_vis.json")))
        self.targets = []

        for p in self.label_paths:
            with open(p, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "visibility" in data:
                    y = data["visibility"]
                elif "label" in data:
                    y = data["label"]
                else:
                    raise ValueError(f"unknown visibility label format: {p}")
            else:
                y = data
            self.targets.append(int(y))

    def __len__(self):
        return len(self.label_paths)

    def get_class_counts(self):
        return Counter(self.targets)

    def __getitem__(self, idx):
        label_path = self.label_paths[idx]
        base = os.path.basename(label_path).replace("_vis.json", "")
        img_path = os.path.join(self.img_root, base + ".jpg")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(self.targets[idx], dtype=torch.long)
        return img, label


class BDDDDrivable(Dataset):
    # drivable segmentation labels from bdd100k_drivable_maps/labels/{split}
    # returns image + segmentation mask for drivable head

    def __init__(self, split, transform=None, mask_resize=True):
        assert split in {"train", "val"}
        self.bdd_root = get_bdd_root()
        self.img_root = resolve_img_root_for_split(self.bdd_root, split)
        self.mask_root = resolve_drivable_root_for_split(self.bdd_root, split)

        self.transform = transform if transform is not None else get_default_transform()
        self.mask_resize = get_default_mask_resize() if mask_resize else None

        self.mask_paths = sorted(glob(os.path.join(self.mask_root, "*_drivable_id.png")))
        self.targets = [0 for _ in self.mask_paths]  # placeholder for sampler compatibility

        self.pixel_counter = Counter()
        self.direct_ratios = []
        self.alt_ratios = []
        self.bg_ratios = []

        for p in self.mask_paths:
            arr = np.array(Image.open(p).convert("L"))
            vals, counts = np.unique(arr, return_counts=True)
            total = arr.size
            per = {int(v): int(c) for v, c in zip(vals, counts)}

            for k, v in per.items():
                self.pixel_counter[k] += v

            self.bg_ratios.append(per.get(0, 0) / total)
            self.alt_ratios.append(per.get(1, 0) / total)
            self.direct_ratios.append(per.get(2, 0) / total)

    def __len__(self):
        return len(self.mask_paths)

    def get_pixel_counts(self):
        return Counter(self.pixel_counter)

    def get_mean_pixel_ratios(self):
        n = max(len(self.mask_paths), 1)
        return {
            "background": float(np.mean(self.bg_ratios)) if self.bg_ratios else 0.0,
            "alternative_drivable": float(np.mean(self.alt_ratios)) if self.alt_ratios else 0.0,
            "direct_drivable": float(np.mean(self.direct_ratios)) if self.direct_ratios else 0.0,
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
