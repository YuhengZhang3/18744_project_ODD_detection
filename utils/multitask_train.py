import math
import os
import torch

from utils.checkpoint import save_checkpoint


def load_model_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


def freeze_backbone_stage1(model):
    for p in model.backbone.parameters():
        p.requires_grad = False

    params = []
    for _, head in model.heads.items():
        for p in head.parameters():
            p.requires_grad = True
            params.append(p)
    return params


def build_stage2_param_groups(model, lr_adapter, lr_heads, weight_decay):
    for p in model.backbone.parameters():
        p.requires_grad = False

    for p in model.backbone.adapter.parameters():
        p.requires_grad = True

    for _, head in model.heads.items():
        for p in head.parameters():
            p.requires_grad = True

    adapter_params = [p for p in model.backbone.adapter.parameters() if p.requires_grad]
    head_params = []
    for _, head in model.heads.items():
        head_params.extend([p for p in head.parameters() if p.requires_grad])

    if len(adapter_params) == 0:
        raise RuntimeError("no trainable adapter params found")
    if len(head_params) == 0:
        raise RuntimeError("no trainable head params found")

    param_groups = [
        {"params": adapter_params, "lr": lr_adapter, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_heads, "weight_decay": weight_decay},
    ]
    return param_groups, len(adapter_params), len(head_params)


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

    return correct_time / max(total_time, 1), correct_scene / max(total_scene, 1)


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


def compute_stage1_steps_per_epoch(loaders, ts_ratio, vis_ratio, road_ratio, steps_per_epoch):
    if steps_per_epoch > 0:
        return steps_per_epoch

    total_ratio = ts_ratio + vis_ratio + road_ratio
    base_steps = max(
        len(loaders["train_ts"]),
        math.ceil(len(loaders["train_vis"]) / max(vis_ratio, 1)),
        math.ceil(len(loaders["train_road"]) / max(road_ratio, 1)),
    )
    return base_steps * total_ratio


def build_pattern(ts_ratio, vis_ratio, road_ratio):
    return ["ts"] * ts_ratio + ["vis"] * vis_ratio + ["road"] * road_ratio


def maybe_save_best(state, save_dir, score, best_score):
    os.makedirs(save_dir, exist_ok=True)

    last_path = os.path.join(save_dir, "last.pt")
    best_path = os.path.join(save_dir, "best.pt")

    save_checkpoint(state, last_path)

    is_best = score > best_score
    if is_best:
        save_checkpoint(state, best_path)
        best_score = score

    return best_score, is_best