import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from utils.infer_utils import (
    load_infer_model,
    build_transform,
    _rerank_road_probs_with_aux,
)
from data.bdd_dataset import TIME_CLASSES, SCENE_CLASSES, VIS_CLASSES, DRIVABLE_CLASSES
from data.rscd_dataset import RSCD_CLASSES, ROAD_STATE_CLASSES, ROAD_SEVERITY_CLASSES


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def softmax_info_from_logits(logits_1d, class_names, topk=3):
    probs = F.softmax(logits_1d.unsqueeze(0), dim=1)[0]
    topk = min(topk, probs.numel())
    vals, idxs = torch.topk(probs, k=topk)

    pred_idx = int(torch.argmax(probs).item())
    return {
        "label": class_names[pred_idx],
        "class_id": pred_idx,
        "confidence": float(probs[pred_idx].item()),
        "topk": [
            {
                "label": class_names[int(i.item())],
                "class_id": int(i.item()),
                "confidence": float(v.item()),
            }
            for v, i in zip(vals, idxs)
        ],
        "probabilities": [float(x) for x in probs.detach().cpu().tolist()],
    }


def summarize_drivable_mask(mask_np):
    total = float(mask_np.size)
    out = {}
    for cid, name in enumerate(DRIVABLE_CLASSES):
        ratio = float((mask_np == cid).sum()) / total if total > 0 else 0.0
        out[name] = ratio
    return out


def make_eval_crops(pil_img):
    w, h = pil_img.size
    crops = [pil_img]

    crop_w = int(w * 0.85)
    crop_h = int(h * 0.85)

    x_positions = [
        0,
        max(0, (w - crop_w) // 2),
        max(0, w - crop_w),
    ]
    y = max(0, (h - crop_h) // 2)

    seen = set()
    for x in x_positions:
        box = (x, y, x + crop_w, y + crop_h)
        if box in seen:
            continue
        seen.add(box)
        crops.append(pil_img.crop(box))

    return crops


def load_pipeline(
    ckpt_path="checkpoints_road_coarse_to_fine/best.pt",
    device=None,
    alpha_state=0.35,
    beta_severity=0.10,
    gate_threshold=0.60,
    gate_power=1.5,
    min_mix=0.0,
    road_topk=3,
    cls_topk=3,
):
    model, device, load_meta = load_infer_model(ckpt_path=ckpt_path, device=device)
    transform = build_transform()

    return {
        "model": model,
        "device": device,
        "transform": transform,
        "ckpt_path": ckpt_path,
        "load_meta": load_meta,
        "params": {
            "alpha_state": alpha_state,
            "beta_severity": beta_severity,
            "gate_threshold": gate_threshold,
            "gate_power": gate_power,
            "min_mix": min_mix,
            "road_topk": road_topk,
            "cls_topk": cls_topk,
        },
    }


def infer_single_pil(pil_img, pipeline):
    model = pipeline["model"]
    device = pipeline["device"]
    transform = pipeline["transform"]
    params = pipeline["params"]

    x = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)

    result = {}

    result["time"] = softmax_info_from_logits(out["time"][0], TIME_CLASSES, topk=params["cls_topk"])
    result["scene"] = softmax_info_from_logits(out["scene"][0], SCENE_CLASSES, topk=params["cls_topk"])
    result["visibility"] = softmax_info_from_logits(out["visibility"][0], VIS_CLASSES, topk=params["cls_topk"])

    result["road_condition_direct"] = softmax_info_from_logits(
        out["road_condition"][0], RSCD_CLASSES, topk=params["road_topk"]
    )
    result["road_state"] = softmax_info_from_logits(
        out["road_state"][0], ROAD_STATE_CLASSES, topk=min(params["cls_topk"], len(ROAD_STATE_CLASSES))
    )
    result["road_severity"] = softmax_info_from_logits(
        out["road_severity"][0], ROAD_SEVERITY_CLASSES, topk=min(params["cls_topk"], len(ROAD_SEVERITY_CLASSES))
    )

    drv_pred = out["drivable"].argmax(dim=1)[0].detach().cpu().numpy()
    result["drivable_summary"] = summarize_drivable_mask(drv_pred)

    crops = make_eval_crops(pil_img)
    crop_tensor = torch.stack([transform(im) for im in crops], dim=0).to(device)

    with torch.no_grad():
        crop_out = model(crop_tensor)
        road_probs = F.softmax(crop_out["road_condition"], dim=1)
        state_probs = F.softmax(crop_out["road_state"], dim=1)
        severity_probs = F.softmax(crop_out["road_severity"], dim=1)

        final_probs_per_crop, reranked_probs, mix_values = _rerank_road_probs_with_aux(
            road_probs,
            state_probs,
            severity_probs,
            alpha=params["alpha_state"],
            beta=params["beta_severity"],
            gate_threshold=params["gate_threshold"],
            gate_power=params["gate_power"],
            min_mix=params["min_mix"],
        )

        weights = final_probs_per_crop.max(dim=1).values
        weights = weights / (weights.sum() + 1e-8)
        final_probs = (final_probs_per_crop * weights[:, None]).sum(dim=0)

    topk = min(params["road_topk"], final_probs.numel())
    vals, idxs = torch.topk(final_probs, k=topk)
    pred_idx = int(torch.argmax(final_probs).item())

    result["road_condition_infer"] = {
        "label": RSCD_CLASSES[pred_idx],
        "class_id": pred_idx,
        "confidence": float(final_probs[pred_idx].item()),
        "topk": [
            {
                "label": RSCD_CLASSES[int(i.item())],
                "class_id": int(i.item()),
                "confidence": float(v.item()),
            }
            for v, i in zip(vals, idxs)
        ],
        "probabilities": [float(x) for x in final_probs.detach().cpu().tolist()],
        "num_crops": len(crops),
        "crop_mix_values": [float(x) for x in mix_values.detach().cpu().tolist()],
        "crop_weights": [float(x) for x in weights.detach().cpu().tolist()],
    }

    return result


def infer_single_image_path(image_path, pipeline):
    pil_img = Image.open(image_path).convert("RGB")
    pred = infer_single_pil(pil_img, pipeline)
    return {
        "image_path": str(image_path),
        "image_name": Path(image_path).name,
        "ckpt_path": pipeline["ckpt_path"],
        "device": str(pipeline["device"]),
        "params": pipeline["params"],
        "prediction": pred,
    }


def gather_images(input_path, max_images=0, recursive=False):
    p = Path(input_path)
    if p.is_file():
        return [p]

    if not p.is_dir():
        raise ValueError("input_path is not a file or directory: {}".format(input_path))

    if recursive:
        files = [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXTS]
    else:
        files = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS]

    files = sorted(files)
    if max_images > 0:
        files = files[:max_images]
    return files


def infer_path(input_path, pipeline, max_images=0, recursive=False):
    image_paths = gather_images(input_path, max_images=max_images, recursive=recursive)
    results = [infer_single_image_path(str(p), pipeline) for p in image_paths]
    return results


def save_results_to_dir(results, output_dir, pipeline=None, input_path=None):
    ensure_dir(output_dir)
    per_image_dir = os.path.join(output_dir, "per_image")
    ensure_dir(per_image_dir)

    for item in results:
        stem = Path(item["image_name"]).stem
        with open(os.path.join(per_image_dir, stem + ".json"), "w") as f:
            json.dump(item, f, indent=2)

    summary = {
        "num_images": len(results),
        "input_path": input_path,
        "ckpt_path": pipeline["ckpt_path"] if pipeline is not None else None,
        "device": str(pipeline["device"]) if pipeline is not None else None,
        "load_meta": pipeline["load_meta"] if pipeline is not None else None,
        "params": pipeline["params"] if pipeline is not None else None,
        "results": results,
    }

    with open(os.path.join(output_dir, "results_all.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "run_info.json"), "w") as f:
        json.dump(
            {
                "num_images": len(results),
                "input_path": input_path,
                "ckpt_path": pipeline["ckpt_path"] if pipeline is not None else None,
                "device": str(pipeline["device"]) if pipeline is not None else None,
                "params": pipeline["params"] if pipeline is not None else None,
            },
            f,
            indent=2,
        )

    return {
        "output_dir": output_dir,
        "per_image_dir": per_image_dir,
        "summary_json": os.path.join(output_dir, "results_all.json"),
        "run_info_json": os.path.join(output_dir, "run_info.json"),
    }
