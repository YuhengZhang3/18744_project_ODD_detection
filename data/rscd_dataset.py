# rscd dataset for road condition classification

import os
from glob import glob
from collections import Counter

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


RSCD_CLASSES = [
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

RSCD_CLASS_TO_IDX = {name: i for i, name in enumerate(RSCD_CLASSES)}

ROAD_STATE_CLASSES = ["dry", "wet", "water", "snow", "ice"]
ROAD_SEVERITY_CLASSES = ["none", "smooth", "slight", "severe"]

ROAD_STATE_TO_IDX = {name: i for i, name in enumerate(ROAD_STATE_CLASSES)}
ROAD_SEVERITY_TO_IDX = {name: i for i, name in enumerate(ROAD_SEVERITY_CLASSES)}


def road_condition_name_to_state(name):
    if name == "ice":
        return "ice"
    if name in ["fresh_snow", "melted_snow"]:
        return "snow"
    return name.split("_")[0]


def road_condition_name_to_severity(name):
    if name == "ice":
        return "none"
    if name in ["fresh_snow", "melted_snow"]:
        return "none"

    parts = name.split("_")
    if len(parts) == 2:
        return "none"
    if len(parts) == 3:
        return parts[2]
    raise ValueError("unexpected road label format: {}".format(name))


def normalize_test_label_name(name):
    return name.strip().replace("-", "_")


def load_majority_label_from_txt(txt_path):
    with open(txt_path, "r") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError("empty label txt: {}".format(txt_path))

    labels = [normalize_test_label_name(x) for x in raw.split() if x.strip()]
    labels = [x for x in labels if x in RSCD_CLASS_TO_IDX]

    if len(labels) == 0:
        raise ValueError("no valid labels found in txt: {}".format(txt_path))

    counter = Counter(labels)
    majority_label = counter.most_common(1)[0][0]
    return majority_label


def build_base_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def build_train_transform(image_size):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.92, 1.0),
                ratio=(0.95, 1.05),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.08,
                contrast=0.08,
                saturation=0.05,
                hue=0.01,
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(1.0, 1.0),
            ),
            transforms.ToTensor(),
        ]
    )


class RSCDRoadCondition(Dataset):
    def __init__(self, root, split="train", image_size=336, augment=False):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.augment = augment

        if split == "train" and augment:
            self.transform = build_train_transform(image_size)
        else:
            self.transform = build_base_transform(image_size)

        self.samples = []

        if split == "train":
            split_root = os.path.join(root, "train-set")

            for class_name in RSCD_CLASSES:
                class_dir = os.path.join(split_root, class_name)
                if not os.path.isdir(class_dir):
                    continue

                img_paths = []
                img_paths.extend(glob(os.path.join(class_dir, "*.jpg")))
                img_paths.extend(glob(os.path.join(class_dir, "*.jpeg")))
                img_paths.extend(glob(os.path.join(class_dir, "*.png")))

                for img_path in sorted(img_paths):
                    self.samples.append((img_path, class_name))

        elif split == "test":
            split_root = os.path.join(root, "test-set")
            scene_dirs = sorted(glob(os.path.join(split_root, "*")))

            for scene_dir in scene_dirs:
                if not os.path.isdir(scene_dir):
                    continue

                img_paths = []
                img_paths.extend(glob(os.path.join(scene_dir, "*.jpg")))
                img_paths.extend(glob(os.path.join(scene_dir, "*.jpeg")))
                img_paths.extend(glob(os.path.join(scene_dir, "*.png")))

                for img_path in sorted(img_paths):
                    stem, _ = os.path.splitext(img_path)
                    txt_path = stem + ".txt"
                    if not os.path.exists(txt_path):
                        continue

                    class_name = load_majority_label_from_txt(txt_path)
                    self.samples.append((img_path, class_name))

        else:
            raise ValueError("unsupported split: {}".format(split))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label_idx = RSCD_CLASS_TO_IDX[class_name]
        state_name = road_condition_name_to_state(class_name)
        severity_name = road_condition_name_to_severity(class_name)

        labels = {
            "road_condition": torch.tensor(label_idx, dtype=torch.long),
            "road_state": torch.tensor(ROAD_STATE_TO_IDX[state_name], dtype=torch.long),
            "road_severity": torch.tensor(ROAD_SEVERITY_TO_IDX[severity_name], dtype=torch.long),
        }
        return img, labels
