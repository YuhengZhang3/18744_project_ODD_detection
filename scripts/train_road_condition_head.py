import os
import sys
import argparse

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch

from data.rscd_dataset import RSCDRoadCondition
from models.odd_model import ODDModel
from utils.common import seed_everything, get_device, ensure_dir
from utils.single_head_utils import (
    make_loader,
    freeze_backbone_only,
    save_last_and_best,
)


def eval_road_condition(model, loader, device):
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(imgs)
            preds = out["road_condition"].argmax(dim=1)

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
        "--data_root",
        type=str,
        default="/home/yuhengz3@andrew.cmu.edu/rscd/dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints_road_condition_rscd",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    ensure_dir(args.save_dir)

    train_set = RSCDRoadCondition(root=args.data_root, split="train")
    test_set = RSCDRoadCondition(root=args.data_root, split="test")

    train_loader = make_loader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = ODDModel(freeze_backbone=False).to(device)

    # keep old behavior:
    # freeze backbone + adapter, train only road_condition head
    freeze_backbone_only(model)
    for p in model.heads["road_condition"].parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.heads["road_condition"].parameters(),
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
    print("test size:", len(test_set))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                cls_feat, _ = model.backbone(imgs)

            logits = model.heads["road_condition"](cls_feat)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach())

        scheduler.step()

        train_loss = running_loss / max(len(train_loader), 1)
        test_acc = eval_road_condition(model, test_loader, device)

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train_loss {train_loss:.4f} "
            f"road_condition_acc {test_acc:.4f}"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "road_condition_acc": test_acc,
            "score": test_acc,
            "args": vars(args),
        }

        best_score, is_best = save_last_and_best(
            state=state,
            save_dir=args.save_dir,
            score=test_acc,
            best_score=best_score,
        )
        if is_best:
            print("saved best to", os.path.join(args.save_dir, "best.pt"))


if __name__ == "__main__":
    main()