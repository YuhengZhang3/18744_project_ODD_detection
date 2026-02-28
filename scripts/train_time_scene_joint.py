import os
import sys
import argparse
from time import time

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
from torch.utils.data import DataLoader

from data.bdd_dataset import BDDDTimeScene, collate_time_scene
from models.odd_model import ODDModel
from losses.odd_losses import odd_loss


def make_loader(img_root, label_dir, batch_size, shuffle):
    ds = BDDDTimeScene(img_root=img_root, label_dir=label_dir)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        collate_fn=collate_time_scene,
        pin_memory=True,
    )
    return loader, len(ds)


def eval_one_epoch(model, loader, device):
    model.eval()
    n = 0
    correct_time = 0
    correct_scene = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)

            for k in batch["labels"]:
                batch["labels"][k] = batch["labels"][k].to(device)
                batch["mask"][k] = batch["mask"][k].to(device)
                batch["severity"][k] = batch["severity"][k].to(device)

            out = model(imgs)

            preds_time = out["time"].argmax(dim=1)
            preds_scene = out["scene"].argmax(dim=1)

            correct_time += (preds_time == batch["labels"]["time"]).sum().item()
            correct_scene += (preds_scene == batch["labels"]["scene"]).sum().item()
            n += imgs.size(0)

    acc_time = correct_time / max(n, 1)
    acc_scene = correct_scene / max(n, 1)
    return acc_time, acc_scene


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    args = parser.parse_args()

    train_img_root = "/home/yuhengz3@andrew.cmu.edu/bdd100k/100k_datasets/100k/train"
    train_label_dir = "/home/yuhengz3@andrew.cmu.edu/bdd100k/100k_label/100k/train"

    val_img_root = "/home/yuhengz3@andrew.cmu.edu/bdd100k/100k_datasets/100k/val"
    val_label_dir = "/home/yuhengz3@andrew.cmu.edu/bdd100k/100k_label/100k/val"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    num_epochs = 15
    lr = 3e-4

    train_loader, train_len = make_loader(train_img_root, train_label_dir, batch_size, shuffle=True)
    val_loader, val_len = make_loader(val_img_root, val_label_dir, batch_size, shuffle=False)

    model = ODDModel(freeze_backbone=True).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.05,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=num_epochs,
    )

    ckpt_dir = "checkpoints_time_scene"
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
    print("batch_size:", batch_size, "epochs:", num_epochs)

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

            if steps % 50 == 0:
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

        acc_time, acc_scene = eval_one_epoch(model, val_loader, device)
        avg_loss = running_loss / max(steps, 1)
        metric = 0.5 * (acc_time + acc_scene)

        print(
            f"[epoch {epoch+1}/{num_epochs}] "
            f"train_loss {avg_loss:.4f} "
            f"time_acc {acc_time:.3f} "
            f"scene_acc {acc_scene:.3f} "
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
