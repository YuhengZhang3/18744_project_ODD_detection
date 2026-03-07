import os
import sys
import json
import argparse
from collections import defaultdict

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
from torch.utils.data import DataLoader, random_split

from data.bdd_dataset import (
    BDDDTimeScene,
    BDDDVisibility,
    collate_time_scene,
    TIME_CLASSES,
    SCENE_CLASSES,
)
from data.rscd_dataset import RSCDRoadCondition, RSCD_CLASSES
from models.odd_model import ODDModel


VIS_CLASSES = ["poor", "medium", "good"]


def build_model(ckpt_path, device):
    model = ODDModel(freeze_backbone=False).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("loaded ckpt:", ckpt_path)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)
    model.eval()
    return model


def print_class_acc(title, class_names, class_correct, class_total):
    print(f"\n{title}")
    for i, name in enumerate(class_names):
        tot = class_total[i]
        cor = class_correct[i]
        acc = cor / tot if tot > 0 else 0.0
        print(f"  {i:>2d} ({name:<12s}) acc={acc:.3f}  {cor}/{tot}")


def eval_time_scene(model, bdd_root, split, batch_size, num_workers, device):
    img_root = os.path.join(bdd_root, "100k_datasets", "100k", split)
    label_dir = os.path.join(bdd_root, "100k_label", "100k", split)

    ds = BDDDTimeScene(img_root=img_root, label_dir=label_dir)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_time_scene,
        drop_last=False,
    )

    total = 0
    correct_time = 0
    correct_scene = 0

    time_total = [0] * len(TIME_CLASSES)
    time_correct = [0] * len(TIME_CLASSES)
    scene_total = [0] * len(SCENE_CLASSES)
    scene_correct = [0] * len(SCENE_CLASSES)

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device, non_blocking=True)
            gt_time = batch["labels"]["time"].to(device, non_blocking=True)
            gt_scene = batch["labels"]["scene"].to(device, non_blocking=True)

            out = model(imgs)
            pred_time = out["time"].argmax(dim=1)
            pred_scene = out["scene"].argmax(dim=1)

            correct_time += (pred_time == gt_time).sum().item()
            correct_scene += (pred_scene == gt_scene).sum().item()
            total += imgs.size(0)

            for g, p in zip(gt_time.tolist(), pred_time.tolist()):
                time_total[g] += 1
                if g == p:
                    time_correct[g] += 1

            for g, p in zip(gt_scene.tolist(), pred_scene.tolist()):
                scene_total[g] += 1
                if g == p:
                    scene_correct[g] += 1

    print(f"\noverall time acc ({split}): {correct_time / max(total, 1):.4f}")
    print(f"overall scene acc ({split}): {correct_scene / max(total, 1):.4f}")
    print_class_acc("per-class time acc:", TIME_CLASSES, time_correct, time_total)
    print_class_acc("per-class scene acc:", SCENE_CLASSES, scene_correct, scene_total)


def eval_visibility(model, split, batch_size, num_workers, device):
    ds = BDDDVisibility(split=split)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    total = 0
    correct = 0
    cls_total = [0] * len(VIS_CLASSES)
    cls_correct = [0] * len(VIS_CLASSES)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(imgs)
            preds = out["visibility"].argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.numel()

            for g, p in zip(labels.tolist(), preds.tolist()):
                cls_total[g] += 1
                if g == p:
                    cls_correct[g] += 1

    print(f"\noverall visibility acc ({split}): {correct / max(total, 1):.4f}")
    print_class_acc("per-class visibility acc:", VIS_CLASSES, cls_correct, cls_total)


def eval_road_test(model, data_root, batch_size, num_workers, device):
    ds = RSCDRoadCondition(root=data_root, split="test")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    total = 0
    correct = 0
    cls_total = [0] * len(RSCD_CLASSES)
    cls_correct = [0] * len(RSCD_CLASSES)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(imgs)
            preds = out["road_condition"].argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.numel()

            for g, p in zip(labels.tolist(), preds.tolist()):
                cls_total[g] += 1
                if g == p:
                    cls_correct[g] += 1

    print(f"\noverall road_condition acc (test): {correct / max(total, 1):.4f}")
    print_class_acc("per-class road_condition acc:", RSCD_CLASSES, cls_correct, cls_total)


def eval_road_valsplit(model, data_root, batch_size, num_workers, device, seed=42, val_ratio=0.1):
    ds_full = RSCDRoadCondition(root=data_root, split="train")
    n = len(ds_full)
    val_len = max(1, int(val_ratio * n))
    train_len = n - val_len
    gen = torch.Generator().manual_seed(seed)
    _, ds_val = random_split(ds_full, [train_len, val_len], generator=gen)

    loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    total = 0
    correct = 0
    cls_total = [0] * len(RSCD_CLASSES)
    cls_correct = [0] * len(RSCD_CLASSES)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(imgs)
            preds = out["road_condition"].argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.numel()

            for g, p in zip(labels.tolist(), preds.tolist()):
                cls_total[g] += 1
                if g == p:
                    cls_correct[g] += 1

    print(f"\noverall road_condition acc (valsplit): {correct / max(total, 1):.4f}")
    print_class_acc("per-class road_condition acc:", RSCD_CLASSES, cls_correct, cls_total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--bdd_root", type=str, default="/home/yuhengz3@andrew.cmu.edu/bdd100k")
    parser.add_argument("--rscd_root", type=str, default="/home/yuhengz3@andrew.cmu.edu/rscd/dataset")
    parser.add_argument("--bdd_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--road_mode", type=str, default="test", choices=["test", "valsplit"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("bdd_split:", args.bdd_split)
    print("road_mode:", args.road_mode)

    model = build_model(args.ckpt_path, device)

    eval_time_scene(
        model=model,
        bdd_root=args.bdd_root,
        split=args.bdd_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    eval_visibility(
        model=model,
        split=args.bdd_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    if args.road_mode == "test":
        eval_road_test(
            model=model,
            data_root=args.rscd_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
    else:
        eval_road_valsplit(
            model=model,
            data_root=args.rscd_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            seed=args.seed,
            val_ratio=args.val_ratio,
        )


if __name__ == "__main__":
    main()
