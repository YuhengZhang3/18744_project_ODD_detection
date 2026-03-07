import os
import sys
import argparse

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
from torch.utils.data import DataLoader, random_split

from data.rscd_dataset import RSCDRoadCondition, RSCD_CLASSES
from models.odd_model import ODDModel


def evaluate(model, loader, device, class_names):
    model.eval()
    total = 0
    correct = 0

    num_classes = len(class_names)
    class_total = [0 for _ in range(num_classes)]
    class_correct = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(imgs)
            preds = out["road_condition"].argmax(dim=1)

            total += labels.numel()
            correct += (preds == labels).sum().item()

            for y, p in zip(labels.tolist(), preds.tolist()):
                class_total[y] += 1
                if y == p:
                    class_correct[y] += 1

    acc = correct / max(total, 1)
    return acc, class_total, class_correct


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
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    road_full = RSCDRoadCondition(root=args.data_root, split="train")
    n = len(road_full)
    val_len = max(1, int(args.val_ratio * n))
    train_len = n - val_len

    gen = torch.Generator().manual_seed(args.seed)
    _, road_val = random_split(road_full, [train_len, val_len], generator=gen)

    loader = DataLoader(
        road_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print("road full train size:", n)
    print("road val size:", len(road_val))
    print("batches:", len(loader))
    print("seed:", args.seed)
    print("val_ratio:", args.val_ratio)

    model = ODDModel(freeze_backbone=False).to(device)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("loaded ckpt:", args.ckpt_path)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)

    class_names = RSCD_CLASSES
    acc, class_total, class_correct = evaluate(model, loader, device, class_names)

    print("\noverall road_condition acc (val split):", acc)
    print("\nper-class acc (val split):")
    for i, name in enumerate(class_names):
        tot = class_total[i]
        cor = class_correct[i]
        a = cor / tot if tot > 0 else 0.0
        print(f"{i:>2d} ({name:<25s}) acc={a:.3f}  {cor}/{tot}")


if __name__ == "__main__":
    main()
