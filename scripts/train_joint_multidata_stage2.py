import os
import sys
import time
import argparse

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch

from models.odd_model import ODDModel
from losses.odd_losses import odd_loss
from utils.common import seed_everything, get_device, ensure_dir
from utils.multitask_data import (
    build_multitask_datasets,
    build_multitask_loaders,
    cycle_loader,
    move_batch_to_device,
    fill_missing_heads,
)
from utils.multitask_train import (
    load_model_ckpt,
    build_stage2_param_groups,
    eval_time_scene,
    eval_visibility,
    eval_road_condition,
    build_pattern,
    maybe_save_best,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr_adapter", type=float, default=1e-5)
    parser.add_argument("--lr_heads", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--steps_per_epoch", type=int, default=1500)

    parser.add_argument("--ts_ratio", type=int, default=2)
    parser.add_argument("--vis_ratio", type=int, default=1)
    parser.add_argument("--road_ratio", type=int, default=2)

    parser.add_argument("--road_loss_weight", type=float, default=1.5)

    parser.add_argument(
        "--init_ckpt",
        type=str,
        default="checkpoints_multitask_stage1/best.pt",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints_multitask_stage2",
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
    device = get_device()
    ensure_dir(args.save_dir)

    datasets = build_multitask_datasets(
        bdd_root=args.bdd_root,
        rscd_root=args.rscd_root,
        seed=args.seed,
        road_val_ratio=0.1,
    )
    loaders = build_multitask_loaders(
        datasets=datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = ODDModel(freeze_backbone=False).to(device)

    missing, unexpected = load_model_ckpt(model, args.init_ckpt)
    print("loaded init ckpt:", args.init_ckpt)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)

    param_groups, n_adapter, n_heads = build_stage2_param_groups(
        model,
        lr_adapter=args.lr_adapter,
        lr_heads=args.lr_heads,
        weight_decay=args.weight_decay,
    )

    opt = torch.optim.AdamW(param_groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    pattern = build_pattern(args.ts_ratio, args.vis_ratio, args.road_ratio)

    ts_iter = cycle_loader(loaders["train_ts"])
    vis_iter = cycle_loader(loaders["train_vis"])
    road_iter = cycle_loader(loaders["train_road"])

    best_score = -1.0

    print("device:", device)
    print("batch_size:", args.batch_size)
    print("epochs:", args.epochs)
    print("steps_per_epoch:", args.steps_per_epoch)
    print("pattern:", pattern)
    print("lr_adapter:", args.lr_adapter)
    print("lr_heads:", args.lr_heads)
    print("road_loss_weight:", args.road_loss_weight)
    print("trainable adapter params:", n_adapter)
    print("trainable head params:", n_heads)
    print(
        "train sizes:",
        "ts=", len(datasets["ts_train"]),
        "vis=", len(datasets["vis_train"]),
        "road=", len(datasets["road_train"]),
    )
    print(
        "val sizes:",
        "ts=", len(datasets["ts_val"]),
        "vis=", len(datasets["vis_val"]),
        "road=", len(datasets["road_val"]),
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        task_loss = {"ts": 0.0, "vis": 0.0, "road": 0.0}
        task_count = {"ts": 0, "vis": 0, "road": 0}

        for step in range(args.steps_per_epoch):
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

            if tag == "road":
                loss = loss * args.road_loss_weight

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_val = float(loss.detach())
            running_loss += loss_val
            task_loss[tag] += loss_val
            task_count[tag] += 1

            if (step + 1) % 100 == 0 or (step + 1) == args.steps_per_epoch:
                dt = time.time() - t0
                avg_loss = running_loss / (step + 1)
                print(
                    f"epoch {epoch}/{args.epochs} "
                    f"step {step + 1}/{args.steps_per_epoch} "
                    f"loss {avg_loss:.4f} "
                    f"time {dt:.1f}s"
                )
                t0 = time.time()

        sched.step()

        train_loss = running_loss / max(args.steps_per_epoch, 1)
        train_ts_loss = task_loss["ts"] / max(task_count["ts"], 1)
        train_vis_loss = task_loss["vis"] / max(task_count["vis"], 1)
        train_road_loss = task_loss["road"] / max(task_count["road"], 1)

        val_time_acc, val_scene_acc = eval_time_scene(model, loaders["val_ts"], device)
        val_vis_acc = eval_visibility(model, loaders["val_vis"], device)
        val_road_acc = eval_road_condition(model, loaders["val_road"], device)

        score = (
            0.25 * val_time_acc
            + 0.25 * val_scene_acc
            + 0.20 * val_vis_acc
            + 0.30 * val_road_acc
        )

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
            f"lr_adapter {opt.param_groups[0]['lr']:.6f} "
            f"lr_heads {opt.param_groups[1]['lr']:.6f}"
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

        best_score, is_best = maybe_save_best(
            state=state,
            save_dir=args.save_dir,
            score=score,
            best_score=best_score,
        )
        if is_best:
            print("saved best to", os.path.join(args.save_dir, "best.pt"))


if __name__ == "__main__":
    main()