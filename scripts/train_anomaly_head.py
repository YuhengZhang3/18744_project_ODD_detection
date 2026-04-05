import os
import sys
import json
import time
import random
import argparse
from collections import defaultdict

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.odd_model import ODDModel
from utils.checkpoint import load_model_ckpt
from data.anomaly_dataset import AnomalyDataset, ANOMALY_CLASSES, collate_anomaly


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_train_transform():
    return transforms.Compose(
        [
            transforms.Resize((336, 336)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_val_transform():
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


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_anomaly_only(model):
    for p in model.anomaly_adapter.parameters():
        p.requires_grad = True
    for p in model.heads["anomalies"].parameters():
        p.requires_grad = True


def count_trainable_params(model):
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
    return total


def make_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_anomaly,
    )


def accuracy_from_lists(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return correct / len(y_true)


def per_class_stats(y_true, y_pred, class_names):
    rows = []
    for i, cname in enumerate(class_names):
        cnt = sum(1 for y in y_true if y == i)
        cor = sum(1 for yt, yp in zip(y_true, y_pred) if yt == i and yp == i)
        acc = cor / cnt if cnt > 0 else 0.0
        rows.append(
            {
                "class_id": i,
                "class_name": cname,
                "count": cnt,
                "correct": cor,
                "acc": acc,
            }
        )
    return rows


def format_per_class(rows, title):
    lines = [title]
    for r in rows:
        lines.append(
            "  {:>2d} ({:<24s}) acc={:.3f}  {}/{}".format(
                r["class_id"],
                r["class_name"],
                r["acc"],
                r["correct"],
                r["count"],
            )
        )
    return "\n".join(lines)


def save_ckpt(model, path, epoch, score, args):
    ensure_dir(os.path.dirname(path))

    # main checkpoint: pure state_dict, compatible with project loaders
    torch.save(model.state_dict(), path)

    # sidecar metadata checkpoint: for resume/debug
    meta_path = path.replace(".pt", "_meta.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "score": score,
            "args": vars(args),
        },
        meta_path,
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    y_true = []
    y_pred = []

    for batch in loader:
        imgs = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"]["anomalies"].to(device, non_blocking=True)

        optimizer.zero_grad()

        out = model(imgs)
        logits = out["anomalies"]

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

        pred = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0
    acc = accuracy_from_lists(y_true, y_pred)
    rows = per_class_stats(y_true, y_pred, ANOMALY_CLASSES)

    return {
        "loss": avg_loss,
        "acc": acc,
        "per_class": rows,
    }


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    for batch in loader:
        imgs = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"]["anomalies"].to(device, non_blocking=True)

        out = model(imgs)
        logits = out["anomalies"]

        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)

        pred = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0
    acc = accuracy_from_lists(y_true, y_pred)
    rows = per_class_stats(y_true, y_pred, ANOMALY_CLASSES)

    return {
        "loss": avg_loss,
        "acc": acc,
        "per_class": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--anomaly_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints_anomaly_head")
    parser.add_argument("--init_ckpt", type=str, default="checkpoints_road_coarse_to_fine/best.pt")
    parser.add_argument("--resume_ckpt", type=str, default="")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.save_dir)

    device = get_device()

    train_ds = AnomalyDataset(
        root=args.anomaly_root,
        split="train",
        val_ratio=args.val_ratio,
        seed=args.seed,
        transform=build_train_transform(),
    )
    val_ds = AnomalyDataset(
        root=args.anomaly_root,
        split="val",
        val_ratio=args.val_ratio,
        seed=args.seed,
        transform=build_val_transform(),
    )

    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(val_ds, args.eval_batch_size, args.num_workers, shuffle=False)

    model = ODDModel(freeze_backbone=False).to(device)

    start_epoch = 1
    best_acc = -1.0

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_acc = float(ckpt.get("score", -1.0))
        print("resumed from:", args.resume_ckpt)
        print("start_epoch:", start_epoch)
        print("best_acc:", best_acc)
    else:
        missing, unexpected = load_model_ckpt(model, args.init_ckpt)
        print("loaded init ckpt:", args.init_ckpt)
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)

    freeze_all(model)
    unfreeze_anomaly_only(model)

    print("device:", device)
    print("train size:", len(train_ds))
    print("val size:", len(val_ds))
    print("trainable params:", count_trainable_params(model))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = eval_one_epoch(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        print(
            "[epoch {}/{}] train_loss {:.4f} train_acc {:.4f} "
            "val_loss {:.4f} val_acc {:.4f} time {:.1f}s".format(
                epoch,
                args.epochs,
                train_metrics["loss"],
                train_metrics["acc"],
                val_metrics["loss"],
                val_metrics["acc"],
                elapsed,
            )
        )

        print(format_per_class(train_metrics["per_class"], "train per-class"))
        print(format_per_class(val_metrics["per_class"], "val per-class"))

        save_ckpt(
            model=model,
            path=os.path.join(args.save_dir, "last.pt"),
            epoch=epoch,
            score=val_metrics["acc"],
            args=args,
        )

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_ckpt(
                model=model,
                path=os.path.join(args.save_dir, "best.pt"),
                epoch=epoch,
                score=best_acc,
                args=args,
            )
            print("saved best to", os.path.join(args.save_dir, "best.pt"))

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        with open(os.path.join(args.save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        with open(os.path.join(args.save_dir, "best_metrics.json"), "w") as f:
            json.dump(
                {
                    "best_val_acc": best_acc,
                    "last_epoch": epoch,
                },
                f,
                indent=2,
            )

    print("done")
    print("best_val_acc:", best_acc)
    print("best_ckpt:", os.path.join(args.save_dir, "best.pt"))


if __name__ == "__main__":
    main()
