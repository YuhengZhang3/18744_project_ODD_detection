import os
import sys
import time
import math
import random
import argparse

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
from torch.utils.data import DataLoader, random_split

from data.bdd_dataset import (
    BDDDTimeScene,
    BDDDVisibility,
    collate_time_scene,
)
from data.rscd_dataset import RSCDRoadCondition
from models.odd_model import ODDModel
from losses.odd_losses import odd_loss


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_visibility(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    labels = torch.stack([x[1] for x in batch], dim=0)
    bsz = labels.shape[0]
    return {
        "images": imgs,
        "labels": {
            "visibility": labels,
        },
        "mask": {
            "visibility": torch.ones(bsz, dtype=torch.float32),
        },
        "severity": {
            "visibility": torch.ones(bsz, dtype=torch.float32),
        },
    }


def collate_road_condition(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    labels = torch.stack([x[1] for x in batch], dim=0)
    bsz = labels.shape[0]
    return {
        "images": imgs,
        "labels": {
            "road_condition": labels,
        },
        "mask": {
            "road_condition": torch.ones(bsz, dtype=torch.float32),
        },
        "severity": {
            "road_condition": torch.ones(bsz, dtype=torch.float32),
        },
    }


def move_batch_to_device(batch, device):
    batch["images"] = batch["images"].to(device, non_blocking=True)
    for group in ["labels", "mask", "severity"]:
        for k in batch[group]:
            batch[group][k] = batch[group][k].to(device, non_blocking=True)
    return batch


ALL_HEADS = ["time", "scene", "visibility", "road_condition"]


def fill_missing_heads(batch):
    device = batch["images"].device
    bsz = batch["images"].shape[0]

    for name in ALL_HEADS:
        if name not in batch["labels"]:
            batch["labels"][name] = torch.zeros(bsz, dtype=torch.long, device=device)
        if name not in batch["mask"]:
            batch["mask"][name] = torch.zeros(bsz, dtype=torch.float32, device=device)
        if name not in batch["severity"]:
            batch["severity"][name] = torch.ones(bsz, dtype=torch.float32, device=device)

    return batch


def freeze_backbone_stage1(model):
    for p in model.backbone.parameters():
        p.requires_grad = False

    params = []
    for _, head in model.heads.items():
        for p in head.parameters():
            p.requires_grad = True
            params.append(p)
    return params


def eval_time_scene(model, loader, device):
    model.eval()
    total_time = 0
    correct_time = 0
    total_scene = 0
    correct_scene = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device, non_blocking=True)
            out = model(imgs)

            y_time = batch["labels"]["time"].to(device, non_blocking=True)
            y_scene = batch["labels"]["scene"].to(device, non_blocking=True)

            p_time = out["time"].argmax(dim=1)
            p_scene = out["scene"].argmax(dim=1)

            total_time += y_time.numel()
            correct_time += (p_time == y_time).sum().item()
            total_scene += y_scene.numel()
            correct_scene += (p_scene == y_scene).sum().item()

    return (
        correct_time / max(total_time, 1),
        correct_scene / max(total_scene, 1),
    )


def eval_visibility(model, loader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device, non_blocking=True)
            out = model(imgs)

            y = batch["labels"]["visibility"].to(device, non_blocking=True)
            p = out["visibility"].argmax(dim=1)

            total += y.numel()
            correct += (p == y).sum().item()

    return correct / max(total, 1)


def eval_road_condition(model, loader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device, non_blocking=True)
            out = model(imgs)

            y = batch["labels"]["road_condition"].to(device, non_blocking=True)
            p = out["road_condition"].argmax(dim=1)

            total += y.numel()
            correct += (p == y).sum().item()

    return correct / max(total, 1)


def make_loader(dataset, batch_size, shuffle, num_workers, collate_fn=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--ts_ratio", type=int, default=2)
    parser.add_argument("--vis_ratio", type=int, default=1)
    parser.add_argument("--road_ratio", type=int, default=1)
    parser.add_argument(
        "--merged_ckpt",
        type=str,
        default="checkpoints_merged/odd_merged_heads.pt",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints_multitask_stage1",
    )
    parser.add_argument(
        "--bdd_root",
        type=str,
        default="/home/yuhengz3@andrew.cmu.edu/bdd100k",
    )
    parser.add_argument(
        "--rscd_root",
        type=str,
        default="/home/yuhengz3@andrew.cmu.edu/rscd/dataset",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # bdd time/scene
    ts_train = BDDDTimeScene(
        img_root=os.path.join(args.bdd_root, "100k_datasets", "100k", "train"),
        label_dir=os.path.join(args.bdd_root, "100k_label", "100k", "train"),
    )
    ts_val = BDDDTimeScene(
        img_root=os.path.join(args.bdd_root, "100k_datasets", "100k", "val"),
        label_dir=os.path.join(args.bdd_root, "100k_label", "100k", "val"),
    )

    # bdd visibility
    vis_train = BDDDVisibility(split="train")
    vis_val = BDDDVisibility(split="val")

    # rscd road condition
    road_full = RSCDRoadCondition(root=args.rscd_root, split="train")
    n = len(road_full)
    val_len = max(1, int(0.1 * n))
    train_len = n - val_len
    gen = torch.Generator().manual_seed(args.seed)
    road_train, road_val = random_split(road_full, [train_len, val_len], generator=gen)

    train_loader_ts = make_loader(
        ts_train, args.batch_size, True, args.num_workers, collate_time_scene
    )
    val_loader_ts = make_loader(
        ts_val, args.batch_size, False, args.num_workers, collate_time_scene
    )

    train_loader_vis = make_loader(
        vis_train, args.batch_size, True, args.num_workers, collate_visibility
    )
    val_loader_vis = make_loader(
        vis_val, args.batch_size, False, args.num_workers, collate_visibility
    )

    train_loader_road = make_loader(
        road_train, args.batch_size, True, args.num_workers, collate_road_condition
    )
    val_loader_road = make_loader(
        road_val, args.batch_size, False, args.num_workers, collate_road_condition
    )

    model = ODDModel(freeze_backbone=False).to(device)

    ckpt = torch.load(args.merged_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("loaded merged ckpt:", args.merged_ckpt)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)

    params = freeze_backbone_stage1(model)
    if not params:
        raise RuntimeError("no trainable params in stage1")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    if args.steps_per_epoch > 0:
        steps_per_epoch = args.steps_per_epoch
    else:
        total_ratio = args.ts_ratio + args.vis_ratio + args.road_ratio
        base_steps = max(
            len(train_loader_ts),
            math.ceil(len(train_loader_vis) / max(args.vis_ratio, 1)),
            math.ceil(len(train_loader_road) / max(args.road_ratio, 1)),
        )
        steps_per_epoch = base_steps * total_ratio

    pattern = (
        ["ts"] * args.ts_ratio
        + ["vis"] * args.vis_ratio
        + ["road"] * args.road_ratio
    )

    ts_iter = cycle_loader(train_loader_ts)
    vis_iter = cycle_loader(train_loader_vis)
    road_iter = cycle_loader(train_loader_road)

    best_score = -1.0

    print("device:", device)
    print("batch_size:", args.batch_size)
    print("epochs:", args.epochs)
    print("lr:", args.lr)
    print("steps_per_epoch:", steps_per_epoch)
    print("pattern:", pattern)
    print("train sizes:",
          "ts=", len(ts_train),
          "vis=", len(vis_train),
          "road=", len(road_train))
    print("val sizes:",
          "ts=", len(ts_val),
          "vis=", len(vis_val),
          "road=", len(road_val))

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        task_loss = {"ts": 0.0, "vis": 0.0, "road": 0.0}
        task_count = {"ts": 0, "vis": 0, "road": 0}

        for step in range(steps_per_epoch):
            tag = pattern[step % len(pattern)]

            if tag == "ts":
                batch = next(ts_iter)
            elif tag == "vis":
                batch = next(vis_iter)
            else:
                batch = next(road_iter)

            batch = move_batch_to_device(batch, device)
            batch = fill_missing_heads(batch)
            out = model(batch["images"])
            loss = odd_loss(out, batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_val = float(loss.detach())
            running_loss += loss_val
            task_loss[tag] += loss_val
            task_count[tag] += 1

            if (step + 1) % 100 == 0 or (step + 1) == steps_per_epoch:
                dt = time.time() - t0
                avg_loss = running_loss / (step + 1)
                print(
                    f"epoch {epoch}/{args.epochs} "
                    f"step {step + 1}/{steps_per_epoch} "
                    f"loss {avg_loss:.4f} "
                    f"time {dt:.1f}s"
                )
                t0 = time.time()

        sched.step()

        train_loss = running_loss / max(steps_per_epoch, 1)
        train_ts_loss = task_loss["ts"] / max(task_count["ts"], 1)
        train_vis_loss = task_loss["vis"] / max(task_count["vis"], 1)
        train_road_loss = task_loss["road"] / max(task_count["road"], 1)

        val_time_acc, val_scene_acc = eval_time_scene(model, val_loader_ts, device)
        val_vis_acc = eval_visibility(model, val_loader_vis, device)
        val_road_acc = eval_road_condition(model, val_loader_road, device)

        score = 0.25 * val_time_acc + 0.25 * val_scene_acc + 0.25 * val_vis_acc + 0.25 * val_road_acc

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train_loss {train_loss:.4f} "
            f"ts_loss {train_ts_loss:.4f} "
            f"vis_loss {train_vis_loss:.4f} "
            f"road_loss {train_road_loss:.4f} "
            f"time_acc {val_time_acc:.4f} "
            f"scene_acc {val_scene_acc:.4f} "
            f"vis_acc {val_vis_acc:.4f} "
            f"road_acc {val_road_acc:.4f} "
            f"score {score:.4f} "
            f"lr {sched.get_last_lr()[0]:.6f}"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "time_acc": val_time_acc,
            "scene_acc": val_scene_acc,
            "visibility_acc": val_vis_acc,
            "road_condition_acc": val_road_acc,
            "score": score,
            "args": vars(args),
        }

        last_path = os.path.join(args.save_dir, "last.pt")
        torch.save(state, last_path)

        if score > best_score:
            best_score = score
            best_path = os.path.join(args.save_dir, "best.pt")
            torch.save(state, best_path)
            print("saved best to", best_path)


if __name__ == "__main__":
    main()
