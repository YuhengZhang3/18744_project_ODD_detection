# data/rscd_dataset.py

import os
from typing import List, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# class names from train-set folder names
RSCD_CLASSES: List[str] = [
    "dry_asphalt_smooth",
    "dry_asphalt_slight",
    "dry_asphalt_severe",
    "dry_concrete_smooth",
    "dry_concrete_slight",
    "dry_concrete_severe",
    "dry_gravel",
    "dry_mud",
    "fresh_snow",
    "melted_snow",
    "water_asphalt_smooth",
    "water_asphalt_slight",
    "water_asphalt_severe",
    "water_concrete_smooth",
    "water_concrete_slight",
    "water_concrete_severe",
    "water_gravel",
    "water_mud",
    "wet_asphalt_smooth",
    "wet_asphalt_slight",
    "wet_asphalt_severe",
    "wet_concrete_smooth",
    "wet_concrete_slight",
    "wet_concrete_severe",
    "wet_gravel",
    "wet_mud",
    "ice",
]

RSCD_CLASS_TO_ID = {name: i for i, name in enumerate(RSCD_CLASSES)}


def get_default_transform():
    # same style as BDD: resize to 336 and normalize
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


class RSCDRoadCondition(Dataset):
    # RSCD train/test as road_condition head

    def __init__(self, root: str, split: str = "train", transform=None):
        # root should be ~/rscd/dataset
        assert split in {"train", "test"}
        self.split = split
        self.transform = transform if transform is not None else get_default_transform()

        if split == "train":
            base_dir = os.path.join(root, "train-set")
            self.samples: List[Tuple[str, int]] = []
            for cls_name in RSCD_CLASSES:
                cls_dir = os.path.join(base_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    fpath = os.path.join(cls_dir, fname)
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    label_id = RSCD_CLASS_TO_ID[cls_name]
                    self.samples.append((fpath, label_id))
        else:
            base_dir = os.path.join(root, "test-set")
            self.samples = []
            # test-set has subfolders 1..12, each with jpg+txt
            for seq_name in sorted(os.listdir(base_dir)):
                seq_dir = os.path.join(base_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    continue
                for fname in os.listdir(seq_dir):
                    if not fname.lower().endswith(".jpg"):
                        continue
                    img_path = os.path.join(seq_dir, fname)
                    txt_path = os.path.splitext(img_path)[0] + ".txt"
                    if not os.path.isfile(txt_path):
                        continue
                    with open(txt_path, "r") as f:
                        label_name = f.read().strip()
                    if label_name not in RSCD_CLASS_TO_ID:
                        continue
                    label_id = RSCD_CLASS_TO_ID[label_name]
                    self.samples.append((img_path, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(label_id, dtype=torch.long)

        labels = {"road_condition": label}
        mask = {"road_condition": torch.tensor(1.0, dtype=torch.float32)}
        severity = {"road_condition": torch.tensor(1.0, dtype=torch.float32)}

        return {
            "image": img,
            "labels": labels,
            "mask": mask,
            "severity": severity,
        }


def collate_rscd(batch):
    imgs = torch.stack([b["image"] for b in batch], dim=0)

    out_labels = {}
    out_mask = {}
    out_sev = {}

    keys = batch[0]["labels"].keys()
    for k in keys:
        out_labels[k] = torch.stack([b["labels"][k] for b in batch], dim=0)
        out_mask[k] = torch.stack([b["mask"][k] for b in batch], dim=0)
        out_sev[k] = torch.stack([b["severity"][k] for b in batch], dim=0)

    return {
        "images": imgs,
        "labels": out_labels,
        "mask": out_mask,
        "severity": out_sev,
    }