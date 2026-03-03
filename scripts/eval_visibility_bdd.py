import os
import sys
from collections import defaultdict

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
from torch.utils.data import DataLoader

from data.bdd_dataset import BDDDVisibility
from models.odd_model import ODDModel


VIS_CLASSES = ["poor", "medium", "good"]


def eval_visibility(ckpt_path, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = BDDDVisibility(split="val")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    print("val size:", len(ds), "batches:", len(loader))

    model = ODDModel(freeze_backbone=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    n = 0
    correct = 0

    cls_correct = defaultdict(int)
    cls_total = defaultdict(int)

    with torch.no_grad():
        for imgs, vis in loader:
            imgs = imgs.to(device, non_blocking=True)
            vis = vis.to(device, non_blocking=True)

            cls_feat, patch_feat = model.backbone(imgs)
            logits = model.heads["visibility"](cls_feat)
            pred = logits.argmax(dim=1)

            correct += (pred == vis).sum().item()
            n += vis.numel()

            for g, p in zip(vis.tolist(), pred.tolist()):
                cls_total[g] += 1
                if g == p:
                    cls_correct[g] += 1

    overall = correct / max(n, 1)
    print("overall visibility acc:", overall)

    print("\nper-class visibility acc:")
    for i, name in enumerate(VIS_CLASSES):
        tot = cls_total[i]
        cor = cls_correct[i]
        acc = cor / tot if tot > 0 else 0.0
        print(f"  {i} ({name:6s})  acc={acc:.3f}  {cor}/{tot}")


if __name__ == "__main__":
    ckpt_path = "checkpoints_visibility/best.pt"
    eval_visibility(ckpt_path=ckpt_path, batch_size=128)