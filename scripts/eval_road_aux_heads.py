import os
import sys
import json
import argparse
from collections import defaultdict

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from PIL import Image
from tqdm import tqdm

from data.rscd_dataset import (
    RSCDRoadCondition,
    RSCD_CLASSES,
    ROAD_STATE_CLASSES,
    ROAD_SEVERITY_CLASSES,
)
from utils.multitask_data import collate_road_condition
from utils.infer_utils import (
    load_infer_model,
    build_transform,
    _rerank_road_probs_with_aux,
)


def make_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_road_condition,
        drop_last=False,
    )


def update_per_class(stats, y_true, y_pred, class_names):
    for t, p in zip(y_true, y_pred):
        t = int(t)
        p = int(p)
        stats[t]["count"] += 1
        stats[t]["correct"] += int(t == p)
        stats[t]["class_name"] = class_names[t]


def summarize_per_class(stats, class_names):
    rows = []
    total = 0
    correct = 0
    for i, cname in enumerate(class_names):
        cnt = stats[i]["count"]
        cor = stats[i]["correct"]
        acc = cor / cnt if cnt > 0 else 0.0
        rows.append(
            {
                "class_id": i,
                "class_name": cname,
                "count": cnt,
                "correct": cor,
                "acc": acc,
            }
        )
        total += cnt
        correct += cor
    overall = correct / total if total > 0 else 0.0
    return overall, rows


def eval_direct_heads(model, loader, device):
    road_stats = defaultdict(lambda: {"count": 0, "correct": 0, "class_name": ""})
    state_stats = defaultdict(lambda: {"count": 0, "correct": 0, "class_name": ""})
    severity_stats = defaultdict(lambda: {"count": 0, "correct": 0, "class_name": ""})

    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device, non_blocking=True)

            y_road = batch["labels"]["road_condition"].to(device, non_blocking=True)
            y_state = batch["labels"]["road_state"].to(device, non_blocking=True)
            y_severity = batch["labels"]["road_severity"].to(device, non_blocking=True)

            out = model(imgs)

            pred_road = out["road_condition"].argmax(dim=1)
            pred_state = out["road_state"].argmax(dim=1)
            pred_severity = out["road_severity"].argmax(dim=1)

            update_per_class(
                road_stats,
                y_road.detach().cpu().tolist(),
                pred_road.detach().cpu().tolist(),
                RSCD_CLASSES,
            )
            update_per_class(
                state_stats,
                y_state.detach().cpu().tolist(),
                pred_state.detach().cpu().tolist(),
                ROAD_STATE_CLASSES,
            )
            update_per_class(
                severity_stats,
                y_severity.detach().cpu().tolist(),
                pred_severity.detach().cpu().tolist(),
                ROAD_SEVERITY_CLASSES,
            )

    road_overall, road_rows = summarize_per_class(road_stats, RSCD_CLASSES)
    state_overall, state_rows = summarize_per_class(state_stats, ROAD_STATE_CLASSES)
    severity_overall, severity_rows = summarize_per_class(severity_stats, ROAD_SEVERITY_CLASSES)

    return {
        "road_condition_overall": road_overall,
        "road_condition_per_class": road_rows,
        "road_state_overall": state_overall,
        "road_state_per_class": state_rows,
        "road_severity_overall": severity_overall,
        "road_severity_per_class": severity_rows,
    }


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


def infer_style_predict_road(
    model,
    pil_img,
    transform,
    device,
    alpha_state=0.35,
    beta_severity=0.10,
    gate_threshold=0.60,
    gate_power=1.5,
    min_mix=0.0,
):
    crops = make_eval_crops(pil_img)
    x = torch.stack([transform(im) for im in crops], dim=0).to(device)

    with torch.no_grad():
        out = model(x)
        road_probs = F.softmax(out["road_condition"], dim=1)
        state_probs = F.softmax(out["road_state"], dim=1)
        severity_probs = F.softmax(out["road_severity"], dim=1)

        final_probs_per_crop, reranked_probs, mix_values = _rerank_road_probs_with_aux(
            road_probs,
            state_probs,
            severity_probs,
            alpha=alpha_state,
            beta=beta_severity,
            gate_threshold=gate_threshold,
            gate_power=gate_power,
            min_mix=min_mix,
        )

    weights = final_probs_per_crop.max(dim=1).values
    weights = weights / (weights.sum() + 1e-8)

    final_probs = (final_probs_per_crop * weights[:, None]).sum(dim=0)
    pred = int(torch.argmax(final_probs).item())
    return pred, [float(v) for v in final_probs.detach().cpu().tolist()]


def eval_infer_style_road(
    model,
    dataset,
    transform,
    device,
    alpha_state=0.35,
    beta_severity=0.10,
    gate_threshold=0.60,
    gate_power=1.5,
    min_mix=0.0,
    max_samples=0,
):
    road_stats = defaultdict(lambda: {"count": 0, "correct": 0, "class_name": ""})

    if hasattr(dataset, "samples"):
        samples = dataset.samples
    elif hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        samples = [dataset.dataset.samples[i] for i in dataset.indices]
    else:
        raise ValueError("dataset does not expose samples in a supported way")

    if max_samples > 0:
        samples = samples[:max_samples]

    iterator = tqdm(
        samples,
        total=len(samples),
        desc="infer-style road eval",
        leave=False,
    )

    for img_path, class_name in iterator:
        pil_img = Image.open(img_path).convert("RGB")
        pred, _ = infer_style_predict_road(
            model=model,
            pil_img=pil_img,
            transform=transform,
            device=device,
            alpha_state=alpha_state,
            beta_severity=beta_severity,
            gate_threshold=gate_threshold,
            gate_power=gate_power,
            min_mix=min_mix,
        )

        true_id = RSCD_CLASSES.index(class_name)
        update_per_class(
            road_stats,
            [true_id],
            [pred],
            RSCD_CLASSES,
        )

    overall, rows = summarize_per_class(road_stats, RSCD_CLASSES)
    return {
        "infer_style_road_condition_overall": overall,
        "infer_style_road_condition_per_class": rows,
    }


def format_block(title, overall, rows):
    lines = [title, "overall acc: {:.4f}".format(overall)]
    for r in rows:
        lines.append(
            "  {:>2d} ({:<22s}) acc={:.3f}  {}/{}".format(
                r["class_id"], r["class_name"], r["acc"], r["correct"], r["count"]
            )
        )
    return "\n".join(lines)


def run_split(
    split_name,
    dataset_for_direct,
    dataset_for_infer,
    model,
    device,
    transform,
    batch_size,
    num_workers,
    alpha_state,
    beta_severity,
    gate_threshold,
    gate_power,
    min_mix,
    infer_max_samples=0,
):
    loader = make_loader(dataset_for_direct, batch_size=batch_size, num_workers=num_workers)

    direct = eval_direct_heads(model, loader, device)
    infer_style = eval_infer_style_road(
        model=model,
        dataset=dataset_for_infer,
        transform=transform,
        device=device,
        alpha_state=alpha_state,
        beta_severity=beta_severity,
        gate_threshold=gate_threshold,
        gate_power=gate_power,
        min_mix=min_mix,
        max_samples=infer_max_samples,
    )

    lines = []
    lines.append("=== {} ===".format(split_name))
    lines.append("")
    lines.append(format_block(
        "{} direct road_condition".format(split_name),
        direct["road_condition_overall"],
        direct["road_condition_per_class"],
    ))
    lines.append("")
    lines.append(format_block(
        "{} direct road_state".format(split_name),
        direct["road_state_overall"],
        direct["road_state_per_class"],
    ))
    lines.append("")
    lines.append(format_block(
        "{} direct road_severity".format(split_name),
        direct["road_severity_overall"],
        direct["road_severity_per_class"],
    ))
    lines.append("")
    lines.append(format_block(
        "{} infer-style road_condition".format(split_name),
        infer_style["infer_style_road_condition_overall"],
        infer_style["infer_style_road_condition_per_class"],
    ))
    lines.append("")

    out_json = {
        "direct": direct,
        "infer_style": infer_style,
    }
    return "\n".join(lines), out_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--rscd_root", type=str, default="/home/yuhengz3@andrew.cmu.edu/rscd/dataset")
    parser.add_argument("--output_dir", type=str, default="eval_outputs/eval_road_aux_heads")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--road_val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha_state", type=float, default=0.35)
    parser.add_argument("--beta_severity", type=float, default=0.10)
    parser.add_argument("--gate_threshold", type=float, default=0.60)
    parser.add_argument("--gate_power", type=float, default=1.5)
    parser.add_argument("--min_mix", type=float, default=0.0)
    parser.add_argument("--infer_max_samples", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, device, load_meta = load_infer_model(ckpt_path=args.ckpt_path)
    transform = build_transform()

    road_train_full = RSCDRoadCondition(root=args.rscd_root, split="train")
    n = len(road_train_full)
    val_len = max(1, int(args.road_val_ratio * n))
    train_len = n - val_len
    gen = torch.Generator().manual_seed(args.seed)
    road_train, road_val = random_split(road_train_full, [train_len, val_len], generator=gen)

    road_test = RSCDRoadCondition(root=args.rscd_root, split="test")

    val_text, val_json = run_split(
        split_name="VAL",
        dataset_for_direct=road_val,
        dataset_for_infer=road_val,
        model=model,
        device=device,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        alpha_state=args.alpha_state,
        beta_severity=args.beta_severity,
        gate_threshold=args.gate_threshold,
        gate_power=args.gate_power,
        min_mix=args.min_mix,
        infer_max_samples=args.infer_max_samples,
    )

    test_text, test_json = run_split(
        split_name="TEST",
        dataset_for_direct=road_test,
        dataset_for_infer=road_test,
        model=model,
        device=device,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        alpha_state=args.alpha_state,
        beta_severity=args.beta_severity,
        gate_threshold=args.gate_threshold,
        gate_power=args.gate_power,
        min_mix=args.min_mix,
        infer_max_samples=args.infer_max_samples,
    )

    report_text = "\n".join([
        "ckpt: {}".format(args.ckpt_path),
        "device: {}".format(device),
        "alpha_state: {}".format(args.alpha_state),
        "beta_severity: {}".format(args.beta_severity),
        "gate_threshold: {}".format(args.gate_threshold),
        "gate_power: {}".format(args.gate_power),
        "min_mix: {}".format(args.min_mix),
        "",
        val_text,
        "",
        test_text,
    ])

    print(report_text)

    with open(os.path.join(args.output_dir, "road_aux_eval.txt"), "w") as f:
        f.write(report_text)

    with open(os.path.join(args.output_dir, "road_aux_eval.json"), "w") as f:
        json.dump(
            {
                "ckpt_path": args.ckpt_path,
                "device": str(device),
                "alpha_state": args.alpha_state,
                "beta_severity": args.beta_severity,
                "gate_threshold": args.gate_threshold,
                "gate_power": args.gate_power,
                "min_mix": args.min_mix,
                "load_meta": load_meta,
                "val": val_json,
                "test": test_json,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
