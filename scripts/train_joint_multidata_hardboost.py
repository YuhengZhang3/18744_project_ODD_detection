import os
import sys
import time
import argparse

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import torch.nn.functional as F

from models.odd_model import ODDModel
from utils.common import seed_everything, get_device, ensure_dir
from utils.checkpoint import save_last_and_best
from utils.multitask_data import (
    build_v2_datasets,
    build_v2_loaders,
    move_batch_to_device,
    fill_missing_heads,
    cycle_loader,
    counter_to_class_weights,
)
from utils.multitask_train import (
    eval_multiclass_head,
    build_v2_pattern,
    compute_steps_per_epoch,
)
from configs.data_stats import (
    BDD_SCENE_TRAIN_COUNTS,
    BDD_VISIBILITY_TRAIN_COUNTS,
    RSCD_ROAD_TRAIN_COUNTS,
    BDD_TIME_TRAIN_COUNTS,
)

TIME_BOOST = {0: 2.0}
SCENE_BOOST = {0: 1.5}
VIS_BOOST = {1: 1.4}
ROAD_BOOST = {
    7: 1.8,
    10: 2.2,
    13: 2.2,
    16: 2.0,
    17: 2.0,
    18: 2.5,
    19: 2.5,
    21: 2.5,
    22: 2.5,
    24: 2.0,
    25: 2.0,
}


def boosted_ce_loss(logits, y, class_weights=None, hard_boost=None):
    loss = F.cross_entropy(logits, y, reduction="none", weight=class_weights)
    if hard_boost:
        boost = torch.ones_like(loss)
        for cid, mul in hard_boost.items():
            boost = torch.where(y == cid, boost * float(mul), boost)
        loss = loss * boost
    return loss.mean()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=200)

    parser.add_argument("--bdd_root", type=str, default="/home/yuhengz3@andrew.cmu.edu/bdd100k")
    parser.add_argument("--rscd_root", type=str, default="/home/yuhengz3@andrew.cmu.edu/rscd/dataset")
    parser.add_argument("--save_dir", type=str, default="checkpoints_multitask_v2_hardboost")
    parser.add_argument("--init_ckpt", type=str, default="checkpoints_multitask_v2_final/best.pt")
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--sampler_cache_dir", type=str, default="cache/sampler_cache")
    parser.add_argument("--sample_weight_cache_dir", type=str, default="cache/hard_weight_cache")

    parser.add_argument("--lr_main", type=float, default=1e-4)
    parser.add_argument("--lr_road", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--ts_ratio", type=int, default=1)
    parser.add_argument("--vis_ratio", type=int, default=1)
    parser.add_argument("--drv_ratio", type=int, default=0)
    parser.add_argument("--road_ratio", type=int, default=3)

    parser.add_argument("--road_loss_weight", type=float, default=2.5)

    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.save_dir)
    device = get_device()

    datasets = build_v2_datasets(
        bdd_root=args.bdd_root,
        rscd_root=args.rscd_root,
        seed=args.seed,
        road_val_ratio=0.1,
    )

    loaders = build_v2_loaders(
        datasets=datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_balanced_scene=True,
        use_balanced_visibility=True,
        use_balanced_road=True,
        sampler_cache_dir=args.sampler_cache_dir,
        sample_weight_cache_dir=args.sample_weight_cache_dir,
        eval_batch_size=args.eval_batch_size,
    )

    model = ODDModel(freeze_backbone=False).to(device)
    model.backbone.freeze_backbone()

    # freeze drivable head for this stage
    if "drivable" in model.heads:
        for p in model.heads["drivable"].parameters():
            p.requires_grad = False

    start_epoch = 1
    best_score = -1.0

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
    else:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)

    main_params = []
    road_params = []

    for p in model.bdd_adapter.parameters():
        if p.requires_grad:
            main_params.append(p)

    for head_name in ["time", "scene", "visibility"]:
        for p in model.heads[head_name].parameters():
            if p.requires_grad:
                main_params.append(p)

    for p in model.road_adapter.parameters():
        if p.requires_grad:
            road_params.append(p)

    for p in model.heads["road_condition"].parameters():
        if p.requires_grad:
            road_params.append(p)

    opt_main = torch.optim.AdamW(main_params, lr=args.lr_main, weight_decay=args.weight_decay)
    opt_road = torch.optim.AdamW(road_params, lr=args.lr_road, weight_decay=args.weight_decay)

    sched_main = torch.optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=args.epochs)
    sched_road = torch.optim.lr_scheduler.CosineAnnealingLR(opt_road, T_max=args.epochs)

    if args.resume_ckpt:
        if "optimizer_main" in ckpt:
            try:
                opt_main.load_state_dict(ckpt["optimizer_main"])
            except Exception:
                pass
        if "scheduler_main" in ckpt:
            try:
                sched_main.load_state_dict(ckpt["scheduler_main"])
            except Exception:
                pass

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("score", -1.0))

    time_class_weights = counter_to_class_weights(BDD_TIME_TRAIN_COUNTS, 4).to(device)
    scene_class_weights = counter_to_class_weights(BDD_SCENE_TRAIN_COUNTS, 7).to(device)
    vis_class_weights = counter_to_class_weights(BDD_VISIBILITY_TRAIN_COUNTS, 3).to(device)
    road_class_weights = counter_to_class_weights(RSCD_ROAD_TRAIN_COUNTS, 27).to(device)

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
        drv_ratio=max(args.drv_ratio, 1),
        road_ratio=args.road_ratio,
        steps_per_epoch=args.steps_per_epoch,
    )

    ts_iter = cycle_loader(loaders["train_ts"])
    vis_iter = cycle_loader(loaders["train_vis"])
    road_iter = cycle_loader(loaders["train_road"])

    print("device:", device)
    print("steps_per_epoch:", steps_per_epoch)
    print("pattern:", pattern)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()

        run_ts = 0.0
        run_vis = 0.0
        run_road = 0.0
        cnt_ts = 0
        cnt_vis = 0
        cnt_road = 0

        for step in range(steps_per_epoch):
            tag = pattern[step % len(pattern)]

            if tag == "ts":
                batch = next(ts_iter)
            elif tag == "vis":
                batch = next(vis_iter)
            elif tag == "road":
                batch = next(road_iter)
            else:
                continue

            batch = move_batch_to_device(batch, device)
            batch = fill_missing_heads(batch)
            out = model(batch["images"])

            if tag == "ts":
                loss_time = boosted_ce_loss(
                    out["time"],
                    batch["labels"]["time"],
                    class_weights=time_class_weights,
                    hard_boost=TIME_BOOST,
                )
                loss_scene = boosted_ce_loss(
                    out["scene"],
                    batch["labels"]["scene"],
                    class_weights=scene_class_weights,
                    hard_boost=SCENE_BOOST,
                )
                loss = loss_time + loss_scene

                opt_main.zero_grad()
                loss.backward()
                opt_main.step()

                run_ts += float(loss.detach())
                cnt_ts += 1

            elif tag == "vis":
                loss = boosted_ce_loss(
                    out["visibility"],
                    batch["labels"]["visibility"],
                    class_weights=vis_class_weights,
                    hard_boost=VIS_BOOST,
                )

                opt_main.zero_grad()
                loss.backward()
                opt_main.step()

                run_vis += float(loss.detach())
                cnt_vis += 1

            elif tag == "road":
                loss = boosted_ce_loss(
                    out["road_condition"],
                    batch["labels"]["road_condition"],
                    class_weights=road_class_weights,
                    hard_boost=ROAD_BOOST,
                )
                loss = args.road_loss_weight * loss

                opt_road.zero_grad()
                loss.backward()
                opt_road.step()

                run_road += float(loss.detach())
                cnt_road += 1


            if (step + 1) % args.log_interval == 0 or (step + 1) == steps_per_epoch:
                print(
                    "epoch {}/{} step {}/{} ts_loss {:.4f} vis_loss {:.4f} road_loss {:.4f}".format(
                        epoch, args.epochs,
                        step + 1, steps_per_epoch,
                        run_ts / max(cnt_ts, 1),
                        run_vis / max(cnt_vis, 1),
                        run_road / max(cnt_road, 1),
                    )
                )

        sched_main.step()
        sched_road.step()

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

        score = (
            0.20 * val_time_acc
            + 0.20 * val_scene_acc
            + 0.20 * val_vis_acc
            + 0.40 * val_road_acc
        )

        print(
            "[epoch {}/{}] ts_loss {:.4f} vis_loss {:.4f} road_loss {:.4f} "
            "time_acc {:.4f} scene_acc {:.4f} vis_acc {:.4f} road_acc {:.4f} "
            "score {:.4f} lr_main {:.6f} lr_road {:.6f} time {:.1f}s".format(
                epoch, args.epochs,
                run_ts / max(cnt_ts, 1),
                run_vis / max(cnt_vis, 1),
                run_road / max(cnt_road, 1),
                val_time_acc,
                val_scene_acc,
                val_vis_acc,
                val_road_acc,
                score,
                opt_main.param_groups[0]["lr"],
                opt_road.param_groups[0]["lr"],
                time.time() - t0,
            )
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer_main": opt_main.state_dict(),
            "optimizer_road": opt_road.state_dict(),
            "scheduler_main": sched_main.state_dict(),
            "scheduler_road": sched_road.state_dict(),
            "time_acc": val_time_acc,
            "scene_acc": val_scene_acc,
            "visibility_acc": val_vis_acc,
            "road_condition_acc": val_road_acc,
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
