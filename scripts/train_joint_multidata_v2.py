import os
import sys
import time
import argparse

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch

from models.odd_model import ODDModel
from utils.common import seed_everything, get_device, ensure_dir
from utils.checkpoint import load_model_ckpt, save_last_and_best
from utils.multitask_data import (
    build_v2_datasets,
    build_v2_loaders,
    move_batch_to_device,
    fill_missing_heads,
    cycle_loader,
    counter_to_class_weights,
)
from utils.multitask_train import (
    build_v2_param_groups,
    compute_main_multitask_loss,
    compute_drivable_loss,
    eval_multiclass_head,
    eval_drivable_iou,
    build_v2_pattern,
    compute_steps_per_epoch,
)
from configs.data_stats import (
    BDD_SCENE_TRAIN_COUNTS,
    BDD_VISIBILITY_TRAIN_COUNTS,
    RSCD_ROAD_TRAIN_COUNTS,
    BDD_DRIVABLE_PIXEL_COUNTS,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=0)

    parser.add_argument("--bdd_root", type=str, default="/home/yuhengz3@andrew.cmu.edu/bdd100k")
    parser.add_argument("--rscd_root", type=str, default="/home/yuhengz3@andrew.cmu.edu/rscd/dataset")
    parser.add_argument("--save_dir", type=str, default="checkpoints_multitask_v2")
    parser.add_argument("--init_ckpt", type=str, default="")
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--sampler_cache_dir", type=str, default="")

    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_bdd_adapter", type=float, default=5e-5)
    parser.add_argument("--lr_road_adapter", type=float, default=5e-5)
    parser.add_argument("--lr_heads", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--ts_ratio", type=int, default=2)
    parser.add_argument("--vis_ratio", type=int, default=1)
    parser.add_argument("--drv_ratio", type=int, default=1)
    parser.add_argument("--road_ratio", type=int, default=2)

    parser.add_argument("--road_loss_weight", type=float, default=1.5)
    parser.add_argument("--drivable_loss_weight", type=float, default=1.0)

    parser.add_argument("--use_balanced_scene", action="store_true")
    parser.add_argument("--use_balanced_visibility", action="store_true")
    parser.add_argument("--use_balanced_road", action="store_true")

    parser.add_argument("--use_class_weights_scene", action="store_true")
    parser.add_argument("--use_class_weights_visibility", action="store_true")
    parser.add_argument("--use_class_weights_road", action="store_true")
    parser.add_argument("--use_class_weights_drivable", action="store_true")

    parser.add_argument("--max_train_steps_debug", type=int, default=0)
    parser.add_argument("--eval_max_batches", type=int, default=0)

    args = parser.parse_args()

    print("A. start")
    seed_everything(args.seed)
    ensure_dir(args.save_dir)
    device = get_device()
    print("A1. device ready:", device)

    print("B. before build_v2_datasets")
    datasets = build_v2_datasets(
        bdd_root=args.bdd_root,
        rscd_root=args.rscd_root,
        seed=args.seed,
        road_val_ratio=0.1,
    )

    print("C. after build_v2_datasets")
    print("D. before build_v2_loaders")
    loaders = build_v2_loaders(
        datasets=datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_balanced_scene=args.use_balanced_scene,
        use_balanced_visibility=args.use_balanced_visibility,
        use_balanced_road=args.use_balanced_road,
        sampler_cache_dir=args.sampler_cache_dir,
        eval_batch_size=args.eval_batch_size,
    )

    print("E. after build_v2_loaders")
    print("F. before build model")
    model = ODDModel(freeze_backbone=False).to(device)
    print("G. after build model")

    if args.freeze_backbone:
        model.backbone.freeze_backbone()

    start_epoch = 1

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("loaded resume ckpt:", args.resume_ckpt)
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)
    elif args.init_ckpt:
        missing, unexpected = load_model_ckpt(model, args.init_ckpt)
        print("loaded init ckpt:", args.init_ckpt)
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)

    print("H. before build optimizers")
    main_param_groups, drv_param_groups = build_v2_param_groups(
        model,
        lr_backbone=args.lr_backbone,
        lr_bdd_adapter=args.lr_bdd_adapter,
        lr_road_adapter=args.lr_road_adapter,
        lr_heads=args.lr_heads,
        weight_decay=args.weight_decay,
    )

    opt_main = torch.optim.AdamW(main_param_groups)
    opt_drv = torch.optim.AdamW(drv_param_groups)

    sched_main = torch.optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=args.epochs)
    sched_drv = torch.optim.lr_scheduler.CosineAnnealingLR(opt_drv, T_max=args.epochs)

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")

        if "optimizer_main" in ckpt:
            opt_main.load_state_dict(ckpt["optimizer_main"])
        if "optimizer_drv" in ckpt:
            opt_drv.load_state_dict(ckpt["optimizer_drv"])
        if "scheduler_main" in ckpt:
            sched_main.load_state_dict(ckpt["scheduler_main"])
        if "scheduler_drv" in ckpt:
            sched_drv.load_state_dict(ckpt["scheduler_drv"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("score", -1.0))

        print("resumed optimizer/scheduler from:", args.resume_ckpt)
        print("resume start_epoch:", start_epoch)
        print("resume best_score:", best_score)

    print("I. after build optimizers")

    scene_class_weights = None
    visibility_class_weights = None
    road_class_weights = None
    drivable_class_weights = None

    if args.use_class_weights_scene:
        scene_class_weights = counter_to_class_weights(BDD_SCENE_TRAIN_COUNTS, num_classes=7)

    if args.use_class_weights_visibility:
        visibility_class_weights = counter_to_class_weights(BDD_VISIBILITY_TRAIN_COUNTS, num_classes=3)

    if args.use_class_weights_road:
        road_class_weights = counter_to_class_weights(RSCD_ROAD_TRAIN_COUNTS, num_classes=27)

    if args.use_class_weights_drivable:
        drivable_class_weights = counter_to_class_weights(BDD_DRIVABLE_PIXEL_COUNTS, num_classes=3)

    pattern = build_v2_pattern(
        args.ts_ratio,
        args.vis_ratio,
        args.drv_ratio,
        args.road_ratio,
    )

    steps_per_epoch = compute_steps_per_epoch(
        loaders=loaders,
        ts_ratio=args.ts_ratio,
        vis_ratio=args.vis_ratio,
        drv_ratio=args.drv_ratio,
        road_ratio=args.road_ratio,
        steps_per_epoch=args.steps_per_epoch,
    )

    ts_iter = cycle_loader(loaders["train_ts"])
    vis_iter = cycle_loader(loaders["train_vis"])
    drv_iter = cycle_loader(loaders["train_drv"])
    road_iter = cycle_loader(loaders["train_road"])

    best_score = -1.0

    print("device:", device)
    print("batch_size:", args.batch_size)
    print("epochs:", args.epochs)
    print("steps_per_epoch:", steps_per_epoch)
    print("pattern:", pattern)
    print("freeze_backbone:", args.freeze_backbone)
    print("road_loss_weight:", args.road_loss_weight)
    print("drivable_loss_weight:", args.drivable_loss_weight)
    print(
        "train sizes:",
        "ts=", len(datasets["ts_train"]),
        "vis=", len(datasets["vis_train"]),
        "drv=", len(datasets["drv_train"]),
        "road=", len(datasets["road_train"]),
    )
    print(
        "val sizes:",
        "ts=", len(datasets["ts_val"]),
        "vis=", len(datasets["vis_val"]),
        "drv=", len(datasets["drv_val"]),
        "road=", len(datasets["road_val"]),
    )

    print("J. before training loop")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()

        running_main_loss = 0.0
        running_drv_loss = 0.0

        task_main_loss = {
            "ts": 0.0,
            "vis": 0.0,
            "road": 0.0,
        }
        task_main_count = {
            "ts": 0,
            "vis": 0,
            "road": 0,
        }
        drv_count = 0

        epoch_steps = steps_per_epoch
        if args.max_train_steps_debug > 0:
            epoch_steps = min(epoch_steps, args.max_train_steps_debug)

        for step in range(epoch_steps):
            tag = pattern[step % len(pattern)]

            if tag == "ts":
                batch = next(ts_iter)
                is_main_task = True
                is_drv_task = False
            elif tag == "vis":
                batch = next(vis_iter)
                is_main_task = True
                is_drv_task = False
            elif tag == "drv":
                batch = next(drv_iter)
                is_main_task = False
                is_drv_task = True
            elif tag == "road":
                batch = next(road_iter)
                is_main_task = True
                is_drv_task = False
            else:
                raise ValueError(f"unknown tag: {tag}")

            batch = move_batch_to_device(batch, device)
            batch = fill_missing_heads(batch)

            outputs = model(batch["images"])

            if is_main_task:
                main_loss, _ = compute_main_multitask_loss(
                    outputs=outputs,
                    batch=batch,
                    scene_class_weights=scene_class_weights,
                    visibility_class_weights=visibility_class_weights,
                    road_class_weights=road_class_weights,
                    road_loss_weight=args.road_loss_weight,
                )

                opt_main.zero_grad()
                main_loss.backward()
                opt_main.step()

                main_loss_val = float(main_loss.detach())
                running_main_loss += main_loss_val

                if tag == "ts":
                    task_main_loss["ts"] += main_loss_val
                    task_main_count["ts"] += 1
                elif tag == "vis":
                    task_main_loss["vis"] += main_loss_val
                    task_main_count["vis"] += 1
                elif tag == "road":
                    task_main_loss["road"] += main_loss_val
                    task_main_count["road"] += 1

            if is_drv_task:
                drv_loss = compute_drivable_loss(
                    outputs=outputs,
                    batch=batch,
                    drivable_class_weights=drivable_class_weights,
                )

                if drv_loss is not None:
                    drv_loss = args.drivable_loss_weight * drv_loss

                    opt_drv.zero_grad()
                    drv_loss.backward()
                    opt_drv.step()

                    drv_loss_val = float(drv_loss.detach())
                    running_drv_loss += drv_loss_val
                    drv_count += 1

            if (step + 1) % 50 == 0 or (step + 1) == epoch_steps:
                dt = time.time() - t0
                avg_main = running_main_loss / max(1, sum(task_main_count.values()))
                avg_drv = running_drv_loss / max(1, drv_count)
                print(
                    f"epoch {epoch}/{args.epochs} "
                    f"step {step + 1}/{epoch_steps} "
                    f"main_loss {avg_main:.4f} "
                    f"drv_loss {avg_drv:.4f} "
                    f"time {dt:.1f}s"
                )
                t0 = time.time()

        sched_main.step()
        sched_drv.step()

        avg_ts = task_main_loss["ts"] / max(task_main_count["ts"], 1)
        avg_vis = task_main_loss["vis"] / max(task_main_count["vis"], 1)
        avg_road = task_main_loss["road"] / max(task_main_count["road"], 1)
        avg_drv = running_drv_loss / max(drv_count, 1)

        val_time_acc = eval_multiclass_head(
            model, loaders["val_ts"], device, "time", "time", max_batches=args.eval_max_batches
        )
        val_scene_acc = eval_multiclass_head(
            model, loaders["val_ts"], device, "scene", "scene", max_batches=args.eval_max_batches
        )
        val_vis_acc = eval_multiclass_head(
            model, loaders["val_vis"], device, "visibility", "visibility", max_batches=args.eval_max_batches
        )
        val_road_acc = eval_multiclass_head(
            model, loaders["val_road"], device, "road_condition", "road_condition", max_batches=args.eval_max_batches
        )
        val_drv_iou = eval_drivable_iou(
            model, loaders["val_drv"], device, max_batches=args.eval_max_batches
        )

        score = (
            0.25 * val_time_acc
            + 0.25 * val_scene_acc
            + 0.20 * val_vis_acc
            + 0.30 * val_road_acc
        )

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"ts_loss {avg_ts:.4f} "
            f"vis_loss {avg_vis:.4f} "
            f"road_loss {avg_road:.4f} "
            f"drv_loss {avg_drv:.4f} "
            f"time_acc {val_time_acc:.4f} "
            f"scene_acc {val_scene_acc:.4f} "
            f"vis_acc {val_vis_acc:.4f} "
            f"road_acc {val_road_acc:.4f} "
            f"drv_iou {val_drv_iou:.4f} "
            f"score {score:.4f} "
            f"lr_main {opt_main.param_groups[0]['lr']:.6f} "
            f"lr_drv {opt_drv.param_groups[0]['lr']:.6f}"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer_main": opt_main.state_dict(),
            "optimizer_drv": opt_drv.state_dict(),
            "scheduler_main": sched_main.state_dict(),
            "scheduler_drv": sched_drv.state_dict(),
            "time_acc": val_time_acc,
            "scene_acc": val_scene_acc,
            "visibility_acc": val_vis_acc,
            "road_condition_acc": val_road_acc,
            "drivable_iou": val_drv_iou,
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
