import os
import sys
import argparse

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch

from data.bdd_dataset import BDDDTimeScene, collate_time_scene
from models.odd_model import ODDModel
from losses.odd_losses import odd_loss
from utils.common import seed_everything, get_device, ensure_dir
from utils.single_head_utils import (
    make_loader,
    freeze_vit_only,
    move_batch_to_device,
    save_last_and_best,
)
from utils.multitask_data import fill_missing_heads


def eval_time_scene(model, loader, device):
    model.eval()

    total_time = 0
    correct_time = 0
    total_scene = 0
    correct_scene = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)

            out = model(batch["images"])

            gt_time = batch["labels"]["time"]
            gt_scene = batch["labels"]["scene"]

            pred_time = out["time"].argmax(dim=1)
            pred_scene = out["scene"].argmax(dim=1)

            total_time += gt_time.numel()
            correct_time += (pred_time == gt_time).sum().item()

            total_scene += gt_scene.numel()
            correct_scene += (pred_scene == gt_scene).sum().item()

    return (
        correct_time / max(total_time, 1),
        correct_scene / max(total_scene, 1),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--bdd_root",
        type=str,
        default="/home/yuhengz3@andrew.cmu.edu/bdd100k",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints_time_scene",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    ensure_dir(args.save_dir)

    train_set = BDDDTimeScene(
        img_root=os.path.join(args.bdd_root, "100k_datasets", "100k", "train"),
        label_dir=os.path.join(args.bdd_root, "100k_label", "100k", "train"),
    )
    val_set = BDDDTimeScene(
        img_root=os.path.join(args.bdd_root, "100k_datasets", "100k", "val"),
        label_dir=os.path.join(args.bdd_root, "100k_label", "100k", "val"),
    )

    train_loader = make_loader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_time_scene,
    )
    val_loader = make_loader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_time_scene,
    )

    model = ODDModel(freeze_backbone=False).to(device)

    # keep old behavior:
    # freeze vit, but adapter remains trainable
    freeze_vit_only(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    best_score = -1.0

    print("device:", device)
    print("train size:", len(train_set))
    print("val size:", len(val_set))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            batch = fill_missing_heads(batch)

            out = model(batch["images"])
            loss = odd_loss(out, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach())

        scheduler.step()

        train_loss = running_loss / max(len(train_loader), 1)
        val_time_acc, val_scene_acc = eval_time_scene(model, val_loader, device)
        score = 0.5 * val_time_acc + 0.5 * val_scene_acc

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train_loss {train_loss:.4f} "
            f"time_acc {val_time_acc:.4f} "
            f"scene_acc {val_scene_acc:.4f} "
            f"score {score:.4f}"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "time_acc": val_time_acc,
            "scene_acc": val_scene_acc,
            "score": score,
            "args": vars(args),
        }

        best_score, is_best = save_last_and_best(
            state=state,
            save_dir=args.save_dir,
            score=score,
            best_score=best_score,
        )
        if is_best:
            print("saved best to", os.path.join(args.save_dir, "best.pt"))


if __name__ == "__main__":
    main()