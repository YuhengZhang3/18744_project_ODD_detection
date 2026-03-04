# scripts/eval_road_condition_head.py

import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from data.rscd_dataset import RSCDRoadConditionSimple, RSCD_CLASSES
from models.odd_model import ODDModel


def eval_road_condition(data_root, ckpt_path, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = RSCDRoadConditionSimple(root=data_root, split="train")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    print("dataset size:", len(ds), "batches:", len(loader))

    model = ODDModel(freeze_backbone=True).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    total = 0
    correct = 0

    correct_per = defaultdict(int)
    total_per = defaultdict(int)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            cls_feat, patch_feat = model.backbone(imgs)
            logits = model.heads["road_condition"](cls_feat)
            preds = logits.argmax(dim=1)

            total += labels.numel()
            correct += (preds == labels).sum().item()

            for g, p in zip(labels.tolist(), preds.tolist()):
                total_per[g] += 1
                if g == p:
                    correct_per[g] += 1

    overall = correct / max(total, 1)
    print("overall road_condition acc:", overall)

    print("\nper-class acc:")
    for idx, name in enumerate(RSCD_CLASSES):
        tot = total_per[idx]
        cor = correct_per[idx]
        acc = cor / tot if tot > 0 else 0.0
        print(f"  {idx:2d} ({name:25s}) acc={acc:.3f}  {cor}/{tot}")


if __name__ == "__main__":
    data_root = "/home/yuhengz3@andrew.cmu.edu/rscd/dataset"
    ckpt_path = "checkpoints_road_condition_rscd/best.pt"

    eval_road_condition(
        data_root=data_root,
        ckpt_path=ckpt_path,
        batch_size=128,
    )