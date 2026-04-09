import json
import math
import os
from glob import glob

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from models.odd_model import ODDModel
from utils.common import get_device, ensure_dir
from utils.checkpoint import load_model_ckpt
from data.bdd_dataset import TIME_CLASSES, SCENE_CLASSES, VIS_CLASSES
from data.rscd_dataset import (
    RSCD_CLASSES,
    ROAD_STATE_CLASSES,
    ROAD_SEVERITY_CLASSES,
)
from torchvision import transforms


DEFAULT_CKPT_PATH = "checkpoints_road_aux_multitask_finetune/best.pt"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_infer_model(ckpt_path=DEFAULT_CKPT_PATH, device=None):
    if device is None:
        device = get_device()

    model = ODDModel(freeze_backbone=False).to(device)
    missing, unexpected = load_model_ckpt(model, ckpt_path)
    model.eval()

    return model, device, {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def _softmax_info(logits, class_names):
    probs = F.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    conf = float(probs[idx].item())
    return {
        "class_id": idx,
        "class_name": class_names[idx],
        "confidence": conf,
        "probabilities": [float(x) for x in probs.detach().cpu().tolist()],
    }


def _probs_to_info(probs_1d, class_names):
    idx = int(torch.argmax(probs_1d).item())
    conf = float(probs_1d[idx].item())
    return {
        "class_id": idx,
        "class_name": class_names[idx],
        "confidence": conf,
        "probabilities": [float(x) for x in probs_1d.detach().cpu().tolist()],
    }


def _predict_drivable_mask(model, img_tensor):
    with torch.no_grad():
        out = model(img_tensor)
        logits = out["drivable"]
        pred = logits.argmax(dim=1)[0].detach().cpu().numpy()
    return pred, logits


def _parse_road_label_name(name):
    if name == "ice":
        return {
            "state": "ice",
            "severity": "none",
        }

    if name in ["fresh_snow", "melted_snow"]:
        return {
            "state": "snow",
            "severity": "none",
        }

    parts = name.split("_")
    if len(parts) == 2:
        return {
            "state": parts[0],
            "severity": "none",
        }
    if len(parts) == 3:
        return {
            "state": parts[0],
            "severity": parts[2],
        }
    raise ValueError("unexpected road label format: {}".format(name))


ROAD_CLASS_TO_STATE_IDX = []
ROAD_CLASS_TO_SEVERITY_IDX = []
for cname in RSCD_CLASSES:
    parsed = _parse_road_label_name(cname)
    ROAD_CLASS_TO_STATE_IDX.append(ROAD_STATE_CLASSES.index(parsed["state"]))
    ROAD_CLASS_TO_SEVERITY_IDX.append(ROAD_SEVERITY_CLASSES.index(parsed["severity"]))


def _rerank_road_probs_with_aux(
    road_probs,
    state_probs,
    severity_probs,
    alpha=0.35,
    beta=0.10,
    gate_threshold=0.60,
    gate_power=1.5,
    min_mix=0.0,
    eps=1e-8,
):
    # road_probs: [B, 27]
    # state_probs: [B, 5]
    # severity_probs: [B, 4]
    # use log-space fusion, then apply uncertainty-aware gating
    B, C = road_probs.shape
    reranked_logits = torch.log(road_probs + eps)

    for c in range(C):
        sidx = ROAD_CLASS_TO_STATE_IDX[c]
        vidx = ROAD_CLASS_TO_SEVERITY_IDX[c]
        reranked_logits[:, c] = (
            reranked_logits[:, c]
            + alpha * torch.log(state_probs[:, sidx] + eps)
            + beta * torch.log(severity_probs[:, vidx] + eps)
        )

    reranked_probs = F.softmax(reranked_logits, dim=1)

    road_conf = road_probs.max(dim=1).values
    denom = max(1e-6, 1.0 - gate_threshold)
    mix = ((1.0 - road_conf) / denom).clamp(min=0.0, max=1.0)
    mix = torch.pow(mix, gate_power)

    if min_mix > 0.0:
        mix = torch.clamp(mix, min=min_mix, max=1.0)

    final_probs = (1.0 - mix[:, None]) * road_probs + mix[:, None] * reranked_probs
    final_probs = final_probs / (final_probs.sum(dim=1, keepdim=True) + eps)
    return final_probs, reranked_probs, mix


def _scale_boxes_xyxy(boxes, src_size_hw, dst_size_wh):
    src_h, src_w = src_size_hw
    dst_w, dst_h = dst_size_wh

    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)

    out = []
    for x1, y1, x2, y2 in boxes:
        xx1 = int(round(x1 * sx))
        yy1 = int(round(y1 * sy))
        xx2 = int(round(x2 * sx))
        yy2 = int(round(y2 * sy))

        xx1 = max(0, min(dst_w - 1, xx1))
        yy1 = max(0, min(dst_h - 1, yy1))
        xx2 = max(xx1 + 1, min(dst_w, xx2))
        yy2 = max(yy1 + 1, min(dst_h, yy2))

        out.append((xx1, yy1, xx2, yy2))
    return out


def _extract_patch_boxes_from_mask(
    mask,
    num_patches=5,
    patch_ratio=0.20,
    min_y_ratio=0.55,
    min_patch_drivable_ratio=0.30,
):
    H, W = mask.shape

    patch_h = max(32, int(H * patch_ratio))
    patch_w = max(32, int(W * patch_ratio))

    y_cut = int(H * min_y_ratio)

    coords = np.argwhere((mask == 2) & (np.arange(H)[:, None] >= y_cut))

    if len(coords) == 0:
        coords = np.argwhere((mask > 0) & (np.arange(H)[:, None] >= y_cut))

    if len(coords) == 0:
        coords = np.argwhere(mask == 2)

    if len(coords) == 0:
        coords = np.argwhere(mask > 0)

    if len(coords) == 0:
        return []

    xs_all = coords[:, 1]
    x_bins = np.linspace(xs_all.min(), xs_all.max() + 1, num=num_patches + 1)

    centers = []
    for bi in range(num_patches):
        x_left = x_bins[bi]
        x_right = x_bins[bi + 1]
        bucket = coords[(coords[:, 1] >= x_left) & (coords[:, 1] < x_right)]

        if len(bucket) == 0:
            continue

        bucket = bucket[np.argsort(bucket[:, 0])]
        cy, cx = bucket[int(0.75 * (len(bucket) - 1))]
        centers.append((int(cy), int(cx)))

    if len(centers) == 0:
        ys = coords[:, 0]
        xs = coords[:, 1]
        centers = [(
            int(np.percentile(ys, 75)),
            int(np.median(xs)),
        )]

    boxes = []
    for cy, cx in centers:
        y1 = max(0, cy - patch_h // 2)
        x1 = max(0, cx - patch_w // 2)
        y2 = min(H, y1 + patch_h)
        x2 = min(W, x1 + patch_w)

        if y2 - y1 < patch_h:
            y1 = max(0, y2 - patch_h)
        if x2 - x1 < patch_w:
            x1 = max(0, x2 - patch_w)

        patch_mask = mask[y1:y2, x1:x2]
        if patch_mask.size == 0:
            continue

        drv_ratio = float((patch_mask > 0).sum()) / float(patch_mask.size)
        if drv_ratio < min_patch_drivable_ratio:
            continue

        boxes.append((x1, y1, x2, y2))

    uniq = []
    seen = set()
    for b in boxes:
        if b not in seen:
            uniq.append(b)
            seen.add(b)

    if len(uniq) == 0:
        coords_fg = np.argwhere(mask > 0)
        if len(coords_fg) > 0:
            ys = coords_fg[:, 0]
            xs = coords_fg[:, 1]
            cy = int(np.percentile(ys, 80))
            cx = int(np.median(xs))
            y1 = max(0, cy - patch_h // 2)
            x1 = max(0, cx - patch_w // 2)
            y2 = min(H, y1 + patch_h)
            x2 = min(W, x1 + patch_w)
            uniq = [(x1, y1, x2, y2)]

    return uniq


def _road_probs_from_patches(
    model,
    pil_img,
    boxes,
    transform,
    device,
    alpha_state=0.35,
    beta_severity=0.10,
    gate_threshold=0.60,
    gate_power=1.5,
    min_mix=0.0,
):
    if not boxes:
        return None, None, None, []

    patch_tensors = []
    valid_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = box
        patch = pil_img.crop((x1, y1, x2, y2))
        patch_tensors.append(transform(patch))
        valid_boxes.append(box)

    x = torch.stack(patch_tensors, dim=0).to(device)

    with torch.no_grad():
        out = model(x)
        road_logits = out["road_condition"]
        state_logits = out["road_state"]
        severity_logits = out["road_severity"]

        road_probs = F.softmax(road_logits, dim=1)
        state_probs = F.softmax(state_logits, dim=1)
        severity_probs = F.softmax(severity_logits, dim=1)

        final_patch_probs, reranked_probs, mix_values = _rerank_road_probs_with_aux(
            road_probs,
            state_probs,
            severity_probs,
            alpha=alpha_state,
            beta=beta_severity,
            gate_threshold=gate_threshold,
            gate_power=gate_power,
            min_mix=min_mix,
        )

    patch_conf_max = final_patch_probs.max(dim=1).values
    weights = patch_conf_max / (patch_conf_max.sum() + 1e-8)

    mean_road_probs = (final_patch_probs * weights[:, None]).sum(dim=0)
    mean_state_probs = (state_probs * weights[:, None]).sum(dim=0)
    mean_severity_probs = (severity_probs * weights[:, None]).sum(dim=0)

    final_road = _probs_to_info(mean_road_probs, RSCD_CLASSES)
    final_state = _probs_to_info(mean_state_probs, ROAD_STATE_CLASSES)
    final_severity = _probs_to_info(mean_severity_probs, ROAD_SEVERITY_CLASSES)

    patch_preds = []
    for i, box in enumerate(valid_boxes):
        road_info = _probs_to_info(final_patch_probs[i], RSCD_CLASSES)
        state_info = _probs_to_info(state_probs[i], ROAD_STATE_CLASSES)
        severity_info = _probs_to_info(severity_probs[i], ROAD_SEVERITY_CLASSES)

        patch_preds.append(
            {
                "box_xyxy": [int(v) for v in box],
                "class_id": road_info["class_id"],
                "class_name": road_info["class_name"],
                "confidence": road_info["confidence"],
                "vote_weight": float(weights[i].item()),
                "aux_mix": float(mix_values[i].item()),
                "state_id": state_info["class_id"],
                "state_name": state_info["class_name"],
                "state_confidence": state_info["confidence"],
                "severity_id": severity_info["class_id"],
                "severity_name": severity_info["class_name"],
                "severity_confidence": severity_info["confidence"],
                "raw_road_probabilities": [float(x) for x in road_probs[i].detach().cpu().tolist()],
                "reranked_road_probabilities": [float(x) for x in reranked_probs[i].detach().cpu().tolist()],
                "final_road_probabilities": [float(x) for x in final_patch_probs[i].detach().cpu().tolist()],
            }
        )

    return final_road, final_state, final_severity, patch_preds


def predict_single_image(
    model,
    pil_img,
    transform,
    device,
    num_road_patches=5,
    road_patch_ratio=0.20,
    alpha_state=0.35,
    beta_severity=0.10,
    gate_threshold=0.60,
    gate_power=1.5,
    min_mix=0.0,
):
    full_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(full_tensor)

    time_pred = _softmax_info(out["time"], TIME_CLASSES)
    scene_pred = _softmax_info(out["scene"], SCENE_CLASSES)
    vis_pred = _softmax_info(out["visibility"], VIS_CLASSES)

    drv_mask = out["drivable"].argmax(dim=1)[0].detach().cpu().numpy()

    mask_boxes = _extract_patch_boxes_from_mask(
        drv_mask,
        num_patches=num_road_patches,
        patch_ratio=road_patch_ratio,
    )

    boxes = _scale_boxes_xyxy(
        mask_boxes,
        src_size_hw=drv_mask.shape,
        dst_size_wh=pil_img.size,
    )

    road_pred, road_state_pred, road_severity_pred, patch_preds = _road_probs_from_patches(
        model=model,
        pil_img=pil_img,
        boxes=boxes,
        transform=transform,
        device=device,
        alpha_state=alpha_state,
        beta_severity=beta_severity,
        gate_threshold=gate_threshold,
        gate_power=gate_power,
        min_mix=min_mix,
    )

    if road_pred is None:
        road_pred = {
            "class_id": -1,
            "class_name": "unknown",
            "confidence": 0.0,
            "probabilities": [],
        }
        road_state_pred = {
            "class_id": -1,
            "class_name": "unknown",
            "confidence": 0.0,
            "probabilities": [],
        }
        road_severity_pred = {
            "class_id": -1,
            "class_name": "unknown",
            "confidence": 0.0,
            "probabilities": [],
        }

    return {
        "time": time_pred,
        "scene": scene_pred,
        "visibility": vis_pred,
        "road_condition": road_pred,
        "road_state": road_state_pred,
        "road_severity": road_severity_pred,
        "drivable": {
            "num_boxes": len(boxes),
            "boxes_xyxy": [[int(v) for v in b] for b in boxes],
            "mask_boxes_xyxy": [[int(v) for v in b] for b in mask_boxes],
        },
        "road_patch_predictions": patch_preds,
        "road_fusion_meta": {
            "alpha_state": alpha_state,
            "beta_severity": beta_severity,
            "gate_threshold": gate_threshold,
            "gate_power": gate_power,
            "min_mix": min_mix,
        },
    }, drv_mask


def run_predictions(
    source_directory,
    output_json_directory,
    ckpt_path=DEFAULT_CKPT_PATH,
    num_road_patches=5,
    road_patch_ratio=0.20,
    alpha_state=0.35,
    beta_severity=0.10,
    gate_threshold=0.60,
    gate_power=1.5,
    min_mix=0.0,
):
    ensure_dir(output_json_directory)

    model, device, meta = load_infer_model(ckpt_path=ckpt_path)
    transform = build_transform()

    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob(os.path.join(source_directory, "*" + ext)))
    paths = sorted(paths)

    summary = {
        "source_directory": source_directory,
        "output_json_directory": output_json_directory,
        "ckpt_path": ckpt_path,
        "device": str(device),
        "num_images": len(paths),
        "load_meta": meta,
        "alpha_state": alpha_state,
        "beta_severity": beta_severity,
        "gate_threshold": gate_threshold,
        "gate_power": gate_power,
        "min_mix": min_mix,
    }

    for img_path in paths:
        pil_img = Image.open(img_path).convert("RGB")
        pred, _ = predict_single_image(
            model=model,
            pil_img=pil_img,
            transform=transform,
            device=device,
            num_road_patches=num_road_patches,
            road_patch_ratio=road_patch_ratio,
            alpha_state=alpha_state,
            beta_severity=beta_severity,
            gate_threshold=gate_threshold,
            gate_power=gate_power,
            min_mix=min_mix,
        )

        record = {
            "image_path": img_path,
            "image_name": os.path.basename(img_path),
            "predictions": pred,
        }

        out_name = os.path.splitext(os.path.basename(img_path))[0] + ".json"
        out_path = os.path.join(output_json_directory, out_name)
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

    summary_path = os.path.join(output_json_directory, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
