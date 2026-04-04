import os
import sys
import time
import argparse

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler

from models.odd_model import ODDModel
from data.rscd_dataset import RSCDRoadCondition
from utils.common import seed_everything, get_device, ensure_dir
from utils.checkpoint import save_last_and_best
from utils.multitask_data import collate_road_condition, counter_to_class_weights
from configs.data_stats import RSCD_ROAD_TRAIN_COUNTS
from configs.hardboost_config import ROAD_HARD_BOOST


def load_cached_sample_weights(cache_path):
    return torch.load(cache_path, map_location="cpu")


def boosted_ce_loss(logits, y, class_weights=None, hard_boost=None):
    loss = F.cross_entropy(logits, y, reduction="none", weight=class_weights)
    if hard_boost:
        boost = torch.ones_like(loss)
        for cid, mul in hard_boost.items():
            boost = torch.where(y == cid, boost * float(mul), boost)
        loss = loss * boost
    return loss.mean()


def make_loader(dataset, batch_size, shuffle, num_workers, collate_fn=None, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )


def eval_road_condition(model, loader, device, max_batches=0):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break

            imgs = batch["images"].to(device, non_blocking=True)
            y = batch["labels"]["road_condition"].to(device, non_blocking=True)

            out = model(imgs)
            pred = out["road_condition"].argmax(dim=1)

            total += y.numel()
            correct += int((pred == y).sum().item())

    return correct / total if total > 0 else 0.0


def freeze_all_except_road_multitask(model):
    for p in model.parameters():
        p.requires_grad = False

    for p in model.road_adapter.parameters():
        p.requires_grad = True

    for head_name in ["road_condition", "road_state", "road_severity"]:
        for p in model.heads[head_name].parameters():
            p.requires_grad = True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)

    parser.add_argument("--rscd_root", type=str, default="/home/yuhengz3@andrew.cmu.edu/rscd/dataset")
    parser.add_argument("--save_dir", type=str, default="checkpoints_road_aux_multitask_finetune_aug")
    parser.add_argument("--init_ckpt", type=str, default="checkpoints_multitask_v2_hardboost/best.pt")
    parser.add_argument("--resume_ckpt", type=str, default="")

    parser.add_argument("--road_val_ratio", type=float, default=0.1)
    parser.add_argument("--sample_weight_cache_dir", type=str, default="cache/hard_weight_cache")

    parser.add_argument("--lr_road", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--lambda_state", type=float, default=0.5)
    parser.add_argument("--lambda_severity", type=float, default=0.3)

    parser.add_argument("--augment", action="store_true")

    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.save_dir)
    device = get_device()

    road_full_train = RSCDRoadCondition(
        root=args.rscd_root,
        split="train",
        augment=args.augment,
    )
    road_full_eval = RSCDRoadCondition(
        root=args.rscd_root,
        split="train",
        augment=False,
    )

    n = len(road_full_train)
    val_len = max(1, int(args.road_val_ratio * n))
    train_len = n - val_len
    gen = torch.Generator().manual_seed(args.seed)

    indices = torch.randperm(n, generator=gen).tolist()
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    from torch.utils.data import Subset
    road_train = Subset(road_full_train, train_indices)
    road_val = Subset(road_full_eval, val_indices)

    sampler = None
    weight_path = os.path.join(args.sample_weight_cache_dir, "rscd_train_sample_weights.pt")
    if os.path.exists(weight_path):
        full_weights = load_cached_sample_weights(weight_path)
        subset_weights = full_weights[train_indices]
        sampler = WeightedRandomSampler(
            weights=subset_weights,
            num_samples=len(subset_weights),
            replacement=True,
        )

    train_loader = make_loader(
        road_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_road_condition,
        sampler=sampler,
    )
    val_loader = make_loader(
        road_val,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_road_condition,
    )

    model = ODDModel(freeze_backbone=False).to(device)
    freeze_all_except_road_multitask(model)

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

    road_params = []
    for p in model.road_adapter.parameters():
        if p.requires_grad:
            road_params.append(p)

    for head_name in ["road_condition", "road_state", "road_severity"]:
        for p in model.heads[head_name].parameters():
            if p.requires_grad:
                road_params.append(p)

    opt_road = torch.optim.AdamW(
        road_params,
        lr=args.lr_road,
        weight_decay=args.weight_decay,
    )
    sched_road = torch.optim.lr_scheduler.CosineAnnealingLR(opt_road, T_max=args.epochs)

    if args.resume_ckpt:
        if "optimizer_road" in ckpt:
            try:
                opt_road.load_state_dict(ckpt["optimizer_road"])
            except Exception:
                pass
        if "scheduler_road" in ckpt:
            try:
                sched_road.load_state_dict(ckpt["scheduler_road"])
            except Exception:
                pass
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("score", -1.0))

    road_class_weights = counter_to_class_weights(RSCD_ROAD_TRAIN_COUNTS, 27).to(device)

    print("device:", device)
    print("train size:", len(road_train))
    print("val size:", len(road_val))
    print("lr_road:", args.lr_road)
    print("lambda_state:", args.lambda_state)
    print("lambda_severity:", args.lambda_severity)
    print("augment:", bool(args.augment))
    print("resume:", bool(args.resume_ckpt))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()

        running_loss = 0.0
        running_steps = 0
        running_road = 0.0
        running_state = 0.0
        running_severity = 0.0

        for step, batch in enumerate(train_loader, start=1):
            imgs = batch["images"].to(device, non_blocking=True)
            y_road = batch["labels"]["road_condition"].to(device, non_blocking=True)
            y_state = batch["labels"]["road_state"].to(device, non_blocking=True)
            y_severity = batch["labels"]["road_severity"].to(device, non_blocking=True)

            out = model(imgs)

            loss_road = boosted_ce_loss(
                out["road_condition"],
                y_road,
                class_weights=road_class_weights,
                hard_boost=ROAD_HARD_BOOST,
            )
            loss_state = F.cross_entropy(out["road_state"], y_state)
            loss_severity = F.cross_entropy(out["road_severity"], y_severity)

            loss = loss_road + args.lambda_state * loss_state + args.lambda_severity * loss_severity

            opt_road.zero_grad()
            loss.backward()
            opt_road.step()

            running_loss += float(loss.detach())
            running_road += float(loss_road.detach())
            running_state += float(loss_state.detach())
            running_severity += float(loss_severity.detach())
            running_steps += 1

            if step % args.log_interval == 0 or step == len(train_loader):
                print(
                    "epoch {}/{} step {}/{} total_loss {:.4f} road_loss {:.4f} state_loss {:.4f} severity_loss {:.4f}".format(
                        epoch, args.epochs,
                        step, len(train_loader),
                        running_loss / max(running_steps, 1),
                        running_road / max(running_steps, 1),
                        running_state / max(running_steps, 1),
                        running_severity / max(running_steps, 1),
                    )
                )

        sched_road.step()

        val_road_acc = eval_road_condition(
            model=model,
            loader=val_loader,
            device=device,
            max_batches=args.eval_max_batches,
        )

        avg_loss = running_loss / max(running_steps, 1)

        print(
            "[epoch {}/{}] total_loss {:.4f} road_loss {:.4f} state_loss {:.4f} severity_loss {:.4f} road_acc {:.4f} lr_road {:.6f} time {:.1f}s".format(
                epoch, args.epochs,
                avg_loss,
                running_road / max(running_steps, 1),
                running_state / max(running_steps, 1),
                running_severity / max(running_steps, 1),
                val_road_acc,
                opt_road.param_groups[0]["lr"],
                time.time() - t0,
            )
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer_road": opt_road.state_dict(),
            "scheduler_road": sched_road.state_dict(),
            "road_condition_acc": val_road_acc,
            "score": val_road_acc,
            "args": vars(args),
        }

        best_score, is_best = save_last_and_best(
            state=state,
            save_dir=args.save_dir,
            score=val_road_acc,
            best_score=best_score,
        )
        if is_best:
            print("saved best to", os.path.join(args.save_dir, "best.pt"))


if __name__ == "__main__":
    main()
