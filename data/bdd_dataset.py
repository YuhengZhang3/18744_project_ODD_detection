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
