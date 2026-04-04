import math
import os
from collections import Counter

import torch
import torch.nn.functional as F

from utils.checkpoint import load_model_ckpt, save_last_and_best


def build_v2_param_groups(
    model,
    lr_backbone,
    lr_bdd_adapter,
    lr_road_adapter,
    lr_heads,
    weight_decay,
):
    params_backbone = []
    params_bdd_adapter = []
    params_road_adapter = []
    params_main_heads = []
    params_drivable_head = []

    for p in model.backbone.parameters():
        if p.requires_grad:
            params_backbone.append(p)

    for p in model.bdd_adapter.parameters():
        if p.requires_grad:
            params_bdd_adapter.append(p)

    for p in model.road_adapter.parameters():
        if p.requires_grad:
            params_road_adapter.append(p)

    for name, head in model.heads.items():
        if name == "drivable":
            for p in head.parameters():
                if p.requires_grad:
                    params_drivable_head.append(p)
        else:
            for p in head.parameters():
                if p.requires_grad:
                    params_main_heads.append(p)

    main_param_groups = []
    if params_backbone:
        main_param_groups.append(
            {"params": params_backbone, "lr": lr_backbone, "weight_decay": weight_decay}
        )
    if params_bdd_adapter:
        main_param_groups.append(
            {"params": params_bdd_adapter, "lr": lr_bdd_adapter, "weight_decay": weight_decay}
        )
    if params_road_adapter:
        main_param_groups.append(
            {"params": params_road_adapter, "lr": lr_road_adapter, "weight_decay": weight_decay}
        )
    if params_main_heads:
        main_param_groups.append(
            {"params": params_main_heads, "lr": lr_heads, "weight_decay": weight_decay}
        )

    if not main_param_groups:
        raise RuntimeError("no main params found for optimizer")

    if not params_drivable_head:
        raise RuntimeError("no drivable head params found for optimizer")

    drv_param_groups = [
        {"params": params_drivable_head, "lr": lr_heads, "weight_decay": weight_decay}
    ]

    return main_param_groups, drv_param_groups


def compute_main_multitask_loss(
    outputs,
    batch,
    scene_class_weights=None,
    visibility_class_weights=None,
    road_class_weights=None,
    road_loss_weight=1.0,
):
    total = 0.0
    stats = {}

    if "time" in outputs and "time" in batch["labels"]:
        y = batch["labels"]["time"]
        m = batch["mask"]["time"]
        if m.sum() > 0:
            loss = F.cross_entropy(outputs["time"], y, reduction="none")
            loss = (loss * m).sum() / (m.sum() + 1e-6)
            total = total + loss
            stats["time"] = float(loss.detach())

    if "scene" in outputs and "scene" in batch["labels"]:
        y = batch["labels"]["scene"]
        m = batch["mask"]["scene"]
        if m.sum() > 0:
            w = scene_class_weights.to(y.device) if scene_class_weights is not None else None
            loss = F.cross_entropy(outputs["scene"], y, reduction="none", weight=w)
            loss = (loss * m).sum() / (m.sum() + 1e-6)
            total = total + loss
            stats["scene"] = float(loss.detach())

    if "visibility" in outputs and "visibility" in batch["labels"]:
        y = batch["labels"]["visibility"]
        m = batch["mask"]["visibility"]
        if m.sum() > 0:
            w = visibility_class_weights.to(y.device) if visibility_class_weights is not None else None
            loss = F.cross_entropy(outputs["visibility"], y, reduction="none", weight=w)
            loss = (loss * m).sum() / (m.sum() + 1e-6)
            total = total + loss
            stats["visibility"] = float(loss.detach())

    if "road_condition" in outputs and "road_condition" in batch["labels"]:
        y = batch["labels"]["road_condition"]
        m = batch["mask"]["road_condition"]
        if m.sum() > 0:
            w = road_class_weights.to(y.device) if road_class_weights is not None else None
            loss = F.cross_entropy(outputs["road_condition"], y, reduction="none", weight=w)
            loss = (loss * m).sum() / (m.sum() + 1e-6)
            loss = road_loss_weight * loss
            total = total + loss
            stats["road_condition"] = float(loss.detach())

    return total, stats


def compute_drivable_loss(outputs, batch, drivable_class_weights=None):
    if "drivable" not in outputs:
        return None
    if "drivable" not in batch["labels"]:
        return None

    logits = outputs["drivable"]
    y = batch["labels"]["drivable"]
    m = batch["mask"]["drivable"]

    if logits.shape[-2:] != y.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=y.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    w = drivable_class_weights.to(y.device) if drivable_class_weights is not None else None
    per = F.cross_entropy(logits, y.long(), reduction="none", weight=w)  # [B, H, W]

    m = m[:, None, None].expand_as(per)
    if m.sum() <= 0:
        return None

    loss = (per * m).sum() / (m.sum() + 1e-6)
    return loss


def eval_multiclass_head(model, loader, device, head_name, label_name, max_batches=0):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break

            imgs = batch["images"].to(device, non_blocking=True)
            y = batch["labels"][label_name].to(device, non_blocking=True)
            out = model(imgs)
            p = out[head_name].argmax(dim=1)

            total += y.numel()
            correct += (p == y).sum().item()

    return correct / max(total, 1)


def eval_drivable_iou(model, loader, device, max_batches=0):
    model.eval()

    inter = 0
    union = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break

            imgs = batch["images"].to(device, non_blocking=True)
            y = batch["labels"]["drivable"].to(device, non_blocking=True)

            out = model(imgs)
            logits = out["drivable"]
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=y.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            pred = logits.argmax(dim=1)

            # foreground iou: classes 1 and 2
            pred_fg = pred > 0
            y_fg = y > 0

            inter += (pred_fg & y_fg).sum().item()
            union += (pred_fg | y_fg).sum().item()

    return inter / max(union, 1)


def build_v2_pattern(ts_ratio, vis_ratio, drv_ratio, road_ratio):
    return (
        ["ts"] * ts_ratio
        + ["vis"] * vis_ratio
        + ["drv"] * drv_ratio
        + ["road"] * road_ratio
    )


def compute_steps_per_epoch(loaders, ts_ratio, vis_ratio, drv_ratio, road_ratio, steps_per_epoch):
    if steps_per_epoch > 0:
        return steps_per_epoch

    total_ratio = ts_ratio + vis_ratio + drv_ratio + road_ratio
    base_steps = max(
        len(loaders["train_ts"]),
        math.ceil(len(loaders["train_vis"]) / max(vis_ratio, 1)),
        math.ceil(len(loaders["train_drv"]) / max(drv_ratio, 1)),
        math.ceil(len(loaders["train_road"]) / max(road_ratio, 1)),
    )
    return base_steps * total_ratio