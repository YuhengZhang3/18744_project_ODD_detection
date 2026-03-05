# scripts/train_road_condition_head.py

import os
import sys
import time
import argparse

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.rscd_dataset import RSCDRoadCondition
from models.odd_model import ODDModel


def freeze_backbone_and_other_heads(model):
    # freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    # freeze all heads except road_condition
    for name, head in model.heads.items():
        if name == "road_condition":
            continue
        for p in head.parameters():
            p.requires_grad = False

    params = []
    if "road_condition" in model.heads:
        for p in model.heads["road_condition"].parameters():
            if p.requires_grad:
                params.append(p)
    return params


def eval_road_condition(model, loader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            cls_feat, _ = model.backbone(imgs)
            logits = model.heads["road_condition"](cls_feat)
            preds = logits.argmax(dim=1)

            total += labels.numel()
            correct += (preds == labels).sum().item()

    acc = correct / max(total, 1)
    return acc


def load_checkpoint(path, model, opt, sched, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if opt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    if sched is not None and "scheduler" in ckpt:
        sched.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_acc = ckpt.get("best_acc", ckpt.get("val_acc", 0.0))
    print(f"resume from {path}, start_epoch={start_epoch}, best_acc={best_acc:.4f}")
    return start_epoch, best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="checkpoint path to resume from, default checkpoints_road_condition_rscd/last.pt",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    num_epochs = 15
    lr = 3e-4

    data_root = "/home/yuhengz3@andrew.cmu.edu/rscd/dataset"

    full_set = RSCDRoadCondition(root=data_root, split="train")
    n = len(full_set)
    val_len = max(1, int(0.1 * n))
    train_len = n - val_len

    # split train and val
    train_set, val_set = random_split(full_set, [train_len, val_len])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model = ODDModel(freeze_backbone=False).to(device)

    params = freeze_backbone_and_other_heads(model)
    if not params:
        raise RuntimeError("no trainable params for road_condition head")

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    ckpt_dir = "checkpoints_road_condition_rscd"
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 1
    best_acc = 0.0

    if args.resume:
        if args.resume_path is None:
            resume_path = os.path.join(ckpt_dir, "last.pt")
        else:
            resume_path = args.resume_path
        if os.path.isfile(resume_path):
            start_epoch, best_acc = load_checkpoint(
                resume_path, model, opt, sched, device
            )
        else:
            print(f"resume path {resume_path} not found, start from scratch")

    print("device:", device)
    print("batch_size:", batch_size, "epochs:", num_epochs)
    print("start_epoch:", start_epoch, "best_acc:", best_acc)

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        steps = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                cls_feat, _ = model.backbone(imgs)

            logits = model.heads["road_condition"](cls_feat)
            loss = crit(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += float(loss.detach())
            steps += 1

            if steps % 100 == 0:
                dt = time.time() - t0
                avg_loss = running_loss / max(steps, 1)
                print(
                    f"epoch {epoch}/{num_epochs} "
                    f"step {steps} "
                    f"loss {avg_loss:.4f} "
                    f"({dt:.1f}s)"
                )
                t0 = time.time()

        sched.step()

        train_loss = running_loss / max(steps, 1)
        val_acc = eval_road_condition(model, val_loader, device)

        print(
            f"[epoch {epoch}/{num_epochs}] "
            f"train_loss {train_loss:.4f} "
            f"val_road_cond_acc {val_acc:.4f} "
            f"lr {sched.get_last_lr()[0]:.6f}"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "val_acc": val_acc,
            "best_acc": best_acc,
        }

        last_path = os.path.join(ckpt_dir, "last.pt")
        torch.save(state, last_path)

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(state, best_path)
            print("saved best to", best_path)


if __name__ == "__main__":
    main()