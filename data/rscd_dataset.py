# data/rscd_dataset.py

import os
from typing import List, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


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
    # root: /home/xxx/rscd/dataset

    def __init__(self, root: str, split: str = "train", transform=None):
        assert split in {"train", "test"}
        self.transform = transform if transform is not None else get_default_transform()
        self.samples: List[Tuple[str, int]] = []

        if split == "train":
            base_dir = os.path.join(root, "train-set")
            for cls_name in RSCD_CLASSES:
                cls_dir = os.path.join(base_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    img_path = os.path.join(cls_dir, fname)
                    label_id = RSCD_CLASS_TO_ID[cls_name]
                    self.samples.append((img_path, label_id))
        else:
            base_dir = os.path.join(root, "test-set")
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(label_id, dtype=torch.long)
        return img, label