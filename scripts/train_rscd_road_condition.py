import os
import sys
import argparse
from time import time

import torch
from torch.utils.data import DataLoader, random_split

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from data.rscd_dataset import RSCDRoadCondition, collate_rscd
from models.odd_model import ODDModel
from losses.odd_losses import odd_loss


def make_loaders(data_root, batch_size, val_ratio=0.1):
    ds_full = RSCDRoadCondition(root=data_root, split="train")
    n = len(ds_full)
    val_len = max(1, int(n * val_ratio))
    train_len = n - val_len

    train_ds, val_ds = random_split(ds_full, [train_len, val_len])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_rscd,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_rscd,
    )

    return train_loader, val_loader, train_len, val_len


def eval_one_epoch(model, loader, device):
    model.eval()
    n = 0
    correct = 0

    num_classes = 27
    correct_per = [0 for _ in range(num_classes)]
    total_per = [0 for _ in range(num_classes)]

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

    acc = correct / max(n, 1)
    per_class = []
    for i in range(num_classes):
        tot = total_per[i]
        cor = correct_per[i]
        if tot > 0:
            per_class.append(cor / tot)
        else:
            per_class.append(0.0)

    return acc, per_class


def save_checkpoint(state, ckpt_dir, name):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, name)
    torch.save(state, path)
    return path


def load_checkpoint(path, model, opt, sched, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if opt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    if sched is not None and "scheduler" in ckpt:
        sched.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_metric = ckpt.get("best_metric", 0.0)
    return start_epoch, best_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/yuhengz3@andrew.cmu.edu/rscd/dataset",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, train_len, val_len = make_loaders(
        args.data_root, args.batch_size, val_ratio=0.1
    )

    model = ODDModel(freeze_backbone=args.freeze_backbone).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=args.epochs,
    )

    ckpt_dir = "checkpoints_road_condition_rscd"
    start_epoch = 0
    best_metric = 0.0

    if args.resume:
        if args.resume_path is None:
            resume_path = os.path.join(ckpt_dir, "last.pt")
        else:
            resume_path = args.resume_path
        if os.path.isfile(resume_path):
            start_epoch, best_metric = load_checkpoint(
                resume_path, model, opt, sched, device
            )
            print("resume from", resume_path, "epoch", start_epoch, "best_metric", best_metric)
        else:
            print("resume flag set but file not found:", resume_path)

    print("device:", device)
    print("train size:", train_len, "val size:", val_len)
    print("batch_size:", args.batch_size, "epochs:", args.epochs)
    print("freeze_backbone:", args.freeze_backbone)

    num_epochs = args.epochs

    for epoch in range(start_epoch, num_epochs):
        model.train()
        t0 = time()
        running_loss = 0.0
        steps = 0

        for batch in train_loader:
            imgs = batch["images"].to(device)

            for k in batch["labels"]:
                batch["labels"][k] = batch["labels"][k].to(device)
                batch["mask"][k] = batch["mask"][k].to(device)
                batch["severity"][k] = batch["severity"][k].to(device)

            out = model(imgs)
            loss = odd_loss(out, batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += float(loss.detach())
            steps += 1

            if steps % 100 == 0:
                dt = time() - t0
                avg_loss = running_loss / steps
                print(
                    f"epoch {epoch+1}/{num_epochs} "
                    f"step {steps} "
                    f"loss {avg_loss:.4f} "
                    f"({dt:.1f}s)"
                )
                t0 = time()

        sched.step()

        val_acc, per_class = eval_one_epoch(model, val_loader, device)
        avg_loss = running_loss / max(steps, 1)
        metric = val_acc

        print(
            f"[epoch {epoch+1}/{num_epochs}] "
            f"train_loss {avg_loss:.4f} "
            f"val_acc {val_acc:.3f} "
            f"metric {metric:.3f} "
            f"lr {sched.get_last_lr()[0]:.6f}"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "best_metric": best_metric,
        }

        last_path = save_checkpoint(state, ckpt_dir, "last.pt")
        print("saved last to", last_path)

        if metric > best_metric:
            best_metric = metric
            best_path = save_checkpoint(state, ckpt_dir, "best.pt")
            print("saved best to", best_path)


if __name__ == "__main__":
    main()