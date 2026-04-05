import os
import random
from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


ANOMALY_CLASSES = [
    "none",
    "extreme_weather",
    "road_blockage_hazard",
    "road_structure_failure",
]

ANOMALY_CLASS_TO_IDX = {name: i for i, name in enumerate(ANOMALY_CLASSES)}


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


class AnomalyDataset(Dataset):
    """
    Simple image-level anomaly classification dataset.

    Expected directory structure:
        root/
          none/
          extreme_weather/
          road_blockage_hazard/
          road_structure_failure/

    split:
        - "train"
        - "val"
        - "all"
    """

    def __init__(
        self,
        root,
        split="train",
        val_ratio=0.2,
        seed=42,
        transform=None,
    ):
        assert split in {"train", "val", "all"}

        self.root = Path(root)
        if not self.root.exists():
            raise RuntimeError(f"anomaly dataset root not found: {root}")

        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        self.transform = transform if transform is not None else get_default_transform()

        self.class_names = ANOMALY_CLASSES
        self.class_to_idx = ANOMALY_CLASS_TO_IDX

        self.items = self._build_items()

    def _list_images(self, class_dir):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted(
            [
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in exts
            ]
        )

    def _build_items(self):
        all_items = []

        for cname in self.class_names:
            cdir = self.root / cname
            if not cdir.exists():
                continue

            label = self.class_to_idx[cname]
            img_paths = self._list_images(cdir)

            class_items = [(str(p), label, cname) for p in img_paths]

            if self.split == "all":
                all_items.extend(class_items)
                continue

            rng = random.Random(self.seed + label)
            rng.shuffle(class_items)

            n_total = len(class_items)
            n_val = int(n_total * self.val_ratio)

            if self.split == "val":
                all_items.extend(class_items[:n_val])
            else:
                all_items.extend(class_items[n_val:])

        return all_items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label, class_name = self.items[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "labels": {
                "anomalies": torch.tensor(label, dtype=torch.long),
            },
            "mask": {
                "anomalies": torch.tensor(1.0, dtype=torch.float32),
            },
            "severity": {
                "anomalies": torch.tensor(1.0, dtype=torch.float32),
            },
            "meta": {
                "image_path": img_path,
                "class_name": class_name,
            },
        }


def collate_anomaly(batch):
    imgs = torch.stack([b["image"] for b in batch], dim=0)

    out_labels = {
        "anomalies": torch.stack([b["labels"]["anomalies"] for b in batch], dim=0)
    }
    out_mask = {
        "anomalies": torch.stack([b["mask"]["anomalies"] for b in batch], dim=0)
    }
    out_severity = {
        "anomalies": torch.stack([b["severity"]["anomalies"] for b in batch], dim=0)
    }

    out_meta = {
        "image_path": [b["meta"]["image_path"] for b in batch],
        "class_name": [b["meta"]["class_name"] for b in batch],
    }

    return {
        "images": imgs,
        "labels": out_labels,
        "mask": out_mask,
        "severity": out_severity,
        "meta": out_meta,
    }
