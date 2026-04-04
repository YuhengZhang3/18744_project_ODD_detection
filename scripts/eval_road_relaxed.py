import os
import sys
import json
import argparse
from collections import defaultdict

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from PIL import Image
from tqdm import tqdm

from data.rscd_dataset import RSCDRoadCondition, RSCD_CLASSES
from utils.infer_utils import load_infer_model, build_transform
from scripts.eval_road_aux_heads import infer_style_predict_road


def parse_label(name):
    if name == "ice":
        return {"state": "ice", "material": "snow_ice", "severity": "none"}

    if name == "fresh_snow":
        return {"state": "snow", "material": "snow_ice", "severity": "none"}

    if name == "melted_snow":
        return {"state": "snow", "material": "snow_ice", "severity": "none"}

    parts = name.split("_")
    if len(parts) == 2:
        return {
            "state": parts[0],
            "material": parts[1],
            "severity": "none",
        }
    if len(parts) == 3:
        return {
            "state": parts[0],
            "material": parts[1],
            "severity": parts[2],
        }

    raise ValueError("unexpected label: {}".format(name))


def is_strict_match(gt_name, pred_name):
    return gt_name == pred_name


def is_relaxed1_match(gt_name, pred_name):
    if gt_name == pred_name:
        return True

    gt = parse_label(gt_name)
    pr = parse_label(pred_name)

    wet_water = {"wet", "water"}

    if (
        gt["state"] in wet_water
        and pr["state"] in wet_water
        and gt["material"] == pr["material"]
        and gt["severity"] == pr["severity"]
    ):
        return True

    return False


def is_relaxed2_match(gt_name, pred_name):
    if is_relaxed1_match(gt_name, pred_name):
        return True

    gt = parse_label(gt_name)
    pr = parse_label(pred_name)

    slight_smooth = {"slight", "smooth"}

    if (
        gt["severity"] in slight_smooth
        and pr["severity"] in slight_smooth
        and gt["state"] == pr["state"]
        and gt["material"] == pr["material"]
    ):
        return True

    return False


def evaluate_dataset(
    model,
    dataset,
    transform,
    device,
    alpha_state,
    beta_severity,
    gate_threshold,
    gate_power,
    min_mix,
):
    strict_total = 0
    strict_correct = 0

    relaxed1_total = 0
    relaxed1_correct = 0

    relaxed2_total = 0
    relaxed2_correct = 0

    per_class = defaultdict(lambda: {
        "count": 0,
        "strict_correct": 0,
        "relaxed1_correct": 0,
        "relaxed2_correct": 0,
    })

    samples = dataset.samples if hasattr(dataset, "samples") else [
        dataset.dataset.samples[i] for i in dataset.indices
    ]

    for img_path, gt_name in tqdm(samples, total=len(samples), desc="road relaxed eval"):
        pil_img = Image.open(img_path).convert("RGB")

        pred_id, pred_probs = infer_style_predict_road(
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
        pred_name = RSCD_CLASSES[pred_id]

        strict_ok = is_strict_match(gt_name, pred_name)
        relaxed1_ok = is_relaxed1_match(gt_name, pred_name)
        relaxed2_ok = is_relaxed2_match(gt_name, pred_name)

        strict_total += 1
        relaxed1_total += 1
        relaxed2_total += 1

        strict_correct += int(strict_ok)
        relaxed1_correct += int(relaxed1_ok)
        relaxed2_correct += int(relaxed2_ok)

        d = per_class[gt_name]
        d["count"] += 1
        d["strict_correct"] += int(strict_ok)
        d["relaxed1_correct"] += int(relaxed1_ok)
        d["relaxed2_correct"] += int(relaxed2_ok)

    strict_acc = strict_correct / strict_total if strict_total > 0 else 0.0
    relaxed1_acc = relaxed1_correct / relaxed1_total if relaxed1_total > 0 else 0.0
    relaxed2_acc = relaxed2_correct / relaxed2_total if relaxed2_total > 0 else 0.0

    rows = []
    for cname in RSCD_CLASSES:
        count = per_class[cname]["count"]
        s = per_class[cname]["strict_correct"]
        r1 = per_class[cname]["relaxed1_correct"]
        r2 = per_class[cname]["relaxed2_correct"]

        rows.append({
            "class_name": cname,
            "count": count,
            "strict_acc": s / count if count > 0 else 0.0,
            "relaxed1_acc": r1 / count if count > 0 else 0.0,
            "relaxed2_acc": r2 / count if count > 0 else 0.0,
            "strict_correct": s,
            "relaxed1_correct": r1,
            "relaxed2_correct": r2,
        })

    return {
        "strict_acc": strict_acc,
        "relaxed1_acc": relaxed1_acc,
        "relaxed2_acc": relaxed2_acc,
        "per_class": rows,
    }


def format_report(title, result):
    lines = [title]
    lines.append("strict exact acc: {:.4f}".format(result["strict_acc"]))
    lines.append("relaxed-1 acc (wet<->water, same material+severity): {:.4f}".format(result["relaxed1_acc"]))
    lines.append("relaxed-2 acc (+ slight<->smooth, same state+material): {:.4f}".format(result["relaxed2_acc"]))
    lines.append("")
    lines.append("per-class:")
    for row in result["per_class"]:
        lines.append(
            "  {:<22s} count={:<4d} strict={:.3f} relaxed1={:.3f} relaxed2={:.3f}".format(
                row["class_name"],
                row["count"],
                row["strict_acc"],
                row["relaxed1_acc"],
                row["relaxed2_acc"],
            )
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--rscd_root", type=str, default="/home/yuhengz3@andrew.cmu.edu/rscd/dataset")
    parser.add_argument("--output_dir", type=str, default="eval_outputs/eval_road_relaxed")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"])
    parser.add_argument("--alpha_state", type=float, default=0.50)
    parser.add_argument("--beta_severity", type=float, default=0.10)
    parser.add_argument("--gate_threshold", type=float, default=0.60)
    parser.add_argument("--gate_power", type=float, default=1.5)
    parser.add_argument("--min_mix", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, device, load_meta = load_infer_model(ckpt_path=args.ckpt_path)
    transform = build_transform()

    dataset = RSCDRoadCondition(root=args.rscd_root, split=args.split)

    result = evaluate_dataset(
        model=model,
        dataset=dataset,
        transform=transform,
        device=device,
        alpha_state=args.alpha_state,
        beta_severity=args.beta_severity,
        gate_threshold=args.gate_threshold,
        gate_power=args.gate_power,
        min_mix=args.min_mix,
    )

    report = []
    report.append("ckpt: {}".format(args.ckpt_path))
    report.append("split: {}".format(args.split))
    report.append("device: {}".format(device))
    report.append("alpha_state: {}".format(args.alpha_state))
    report.append("beta_severity: {}".format(args.beta_severity))
    report.append("gate_threshold: {}".format(args.gate_threshold))
    report.append("gate_power: {}".format(args.gate_power))
    report.append("min_mix: {}".format(args.min_mix))
    report.append("")
    report.append(format_report(args.split.upper(), result))
    report_text = "\n".join(report)

    print(report_text)

    with open(os.path.join(args.output_dir, "road_relaxed_eval.txt"), "w") as f:
        f.write(report_text)

    with open(os.path.join(args.output_dir, "road_relaxed_eval.json"), "w") as f:
        json.dump(
            {
                "ckpt_path": args.ckpt_path,
                "split": args.split,
                "device": str(device),
                "alpha_state": args.alpha_state,
                "beta_severity": args.beta_severity,
                "gate_threshold": args.gate_threshold,
                "gate_power": args.gate_power,
                "min_mix": args.min_mix,
                "load_meta": load_meta,
                "result": result,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
