import os
import json
from glob import glob

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

TIME_MAP = {n: i for i, n in enumerate(TIME_CLASSES)}
SCENE_MAP = {n: i for i, n in enumerate(SCENE_CLASSES)}


def get_default_transform():
    # resize to 336 and normalize
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


def get_bdd_root():
    # try server first, then local mac
    candidates = [
        "/home/yuhengz3@andrew.cmu.edu/bdd100k",
        "/Users/zhangyuheng/Documents/Study/CMU_Courses/2026Spring/18744/Project/datasets",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise RuntimeError("bdd root not found, update get_bdd_root()")


class BDDDTimeScene(Dataset):
    # bdd100k time + scene from per-image json files

    def __init__(self, img_root, label_dir, transform=None):
        self.img_root = img_root
        self.label_paths = sorted(glob(os.path.join(label_dir, "*.json")))
        self.transform = transform if transform is not None else get_default_transform()

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

def resolve_img_root_for_split(root, split):
    cand1 = os.path.join(root, "100k_datasets", split)
    cand2 = os.path.join(root, "100k_datasets", "100k", split)
    for c in (cand1, cand2):
        if os.path.isdir(c):
            return c
    raise RuntimeError(f"no img dir for split {split} under {root}")

class BDDDVisibility(Dataset):
    # visibility labels from *_vis.json

    def __init__(self, split, transform=None):
        self.bdd_root = get_bdd_root()
        self.img_root = resolve_img_root_for_split(self.bdd_root, split)
        self.vis_root = os.path.join(self.bdd_root, "visibility_labels", split)

        if transform is None:
            self.transform = get_default_transform()
        else:
            self.transform = transform

        self.items = []
        files = sorted(glob(os.path.join(self.vis_root, "*_vis.json")))
        for jpath in files:
            with open(jpath, "r") as f:
                info = json.load(f)
            name = info.get("name", None)
            vis = info.get("visibility", None)
            if name is None or vis is None:
                continue
            img_path = os.path.join(self.img_root, name)
            if not os.path.exists(img_path):
                continue
            self.items.append((img_path, int(vis)))

        print(f"visibility {split}: {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, vis = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(vis, dtype=torch.long)
        return img, label