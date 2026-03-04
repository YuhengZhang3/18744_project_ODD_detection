import os
import sys
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from data.rscd_dataset import RSCDRoadCondition, collate_rscd, RSCD_CLASSES
from models.odd_model import ODDModel


def eval_rscd(split, data_root, ckpt_path, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("split:", split)

    ds = RSCDRoadCondition(root=data_root, split="train")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_rscd,
    )
    print("dataset size:", len(ds), "batches:", len(loader))

    model = ODDModel(freeze_backbone=True).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    n = 0
    correct = 0

    correct_per = defaultdict(int)
    total_per = defaultdict(int)

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            labels = batch["labels"]["road_condition"].to(device)

            out = model(imgs)
            logits = out["road_condition"]
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            n += imgs.size(0)

            for g, p in zip(labels.tolist(), preds.tolist()):
                total_per[g] += 1
                if g == p:
                    correct_per[g] += 1

    overall = correct / max(n, 1)
    print("overall road_condition acc:", overall)

    print("\nper-class acc:")
    for idx, name in enumerate(RSCD_CLASSES):
        tot = total_per[idx]
        cor = correct_per[idx]
        acc = cor / tot if tot > 0 else 0.0
        print(f"  {idx:2d} ({name:25s}) acc={acc:.3f}  {cor}/{tot}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/yuhengz3@andrew.cmu.edu/rscd/dataset",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints_road_condition_rscd/best.pt",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    eval_rscd(
        split=args.split,
        data_root=args.data_root,
        ckpt_path=args.ckpt_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()