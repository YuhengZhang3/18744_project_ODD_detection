import os
import sys
import argparse

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch

from data.bdd_dataset import BDDDVisibility
from models.odd_model import ODDModel
from utils.common import seed_everything, get_device, ensure_dir
from utils.single_head_utils import (
    make_loader,
    freeze_backbone_only,
    save_last_and_best,
)


def eval_visibility(model, loader, device):
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(imgs)
            preds = out["visibility"].argmax(dim=1)

            total += labels.numel()
            correct += (preds == labels).sum().item()

    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints_visibility",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    ensure_dir(args.save_dir)

    train_set = BDDDVisibility(split="train")
    val_set = BDDDVisibility(split="val")

    train_loader = make_loader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = ODDModel(freeze_backbone=False).to(device)

    # keep old behavior:
    # freeze backbone + adapter, train only visibility head
    freeze_backbone_only(model)
    for p in model.heads["visibility"].parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.heads["visibility"].parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    criterion = torch.nn.CrossEntropyLoss()
    best_score = -1.0

    print("device:", device)
    print("train size:", len(train_set))
    print("val size:", len(val_set))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                cls_feat, _ = model.backbone(imgs)

            logits = model.heads["visibility"](cls_feat)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach())

        scheduler.step()

        train_loss = running_loss / max(len(train_loader), 1)
        val_acc = eval_visibility(model, val_loader, device)

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train_loss {train_loss:.4f} "
            f"visibility_acc {val_acc:.4f}"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "visibility_acc": val_acc,
            "score": val_acc,
            "args": vars(args),
        }

        best_score, is_best = save_last_and_best(
            state=state,
            save_dir=args.save_dir,
            score=val_acc,
            best_score=best_score,
        )
        if is_best:
            print("saved best to", os.path.join(args.save_dir, "best.pt"))


if __name__ == "__main__":
    main()