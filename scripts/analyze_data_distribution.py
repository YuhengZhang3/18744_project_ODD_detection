import os
import sys
import csv
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import numpy as np
from PIL import Image

from data.bdd_dataset import TIME_CLASSES, SCENE_CLASSES
from data.rscd_dataset import RSCD_CLASSES


VIS_CLASSES = ["poor", "medium", "good"]
DRIVABLE_IDS = {
    0: "background",
    1: "alternative_drivable",
    2: "direct_drivable",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_counter_csv(save_path, title, class_names, counter):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title", title])
        writer.writerow(["class_id", "class_name", "count", "ratio"])
        total = sum(counter.values())
        for i, name in enumerate(class_names):
            cnt = counter.get(i, 0)
            ratio = cnt / total if total > 0 else 0.0
            writer.writerow([i, name, cnt, f"{ratio:.6f}"])


def write_key_counter_csv(save_path, title, key_to_name, counter):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title", title])
        writer.writerow(["class_id", "class_name", "count", "ratio"])
        total = sum(counter.values())
        for k in sorted(key_to_name.keys()):
            cnt = counter.get(k, 0)
            ratio = cnt / total if total > 0 else 0.0
            writer.writerow([k, key_to_name[k], cnt, f"{ratio:.6f}"])


def summarize_counter(title, class_names, counter):
    lines = []
    total = sum(counter.values())
    lines.append(f"{title} (total={total})")
    for i, name in enumerate(class_names):
        cnt = counter.get(i, 0)
        ratio = cnt / total if total > 0 else 0.0
        lines.append(f"  {i:>2d} | {name:<25s} | count={cnt:<8d} ratio={ratio:.4f}")
    return "\n".join(lines)


def summarize_key_counter(title, key_to_name, counter):
    lines = []
    total = sum(counter.values())
    lines.append(f"{title} (total={total})")
    for k in sorted(key_to_name.keys()):
        cnt = counter.get(k, 0)
        ratio = cnt / total if total > 0 else 0.0
        lines.append(f"  {k:>2d} | {key_to_name[k]:<25s} | count={cnt:<8d} ratio={ratio:.4f}")
    return "\n".join(lines)


def analyze_bdd_time_scene_split(label_dir):
    time_counter = Counter()
    scene_counter = Counter()

    time_to_idx = {name: i for i, name in enumerate(TIME_CLASSES)}
    scene_to_idx = {name: i for i, name in enumerate(SCENE_CLASSES)}

    paths = sorted(Path(label_dir).glob("*.json"))
    for p in paths:
        with open(p, "r") as f:
            data = json.load(f)

        attrs = data.get("attributes", {})
        t = attrs.get("timeofday", "undefined")
        s = attrs.get("scene", "undefined")

        if t not in time_to_idx:
            t = "undefined"
        if s not in scene_to_idx:
            s = "undefined"

        time_counter[time_to_idx[t]] += 1
        scene_counter[scene_to_idx[s]] += 1

    return time_counter, scene_counter, len(paths)


def analyze_visibility_split(vis_dir):
    counter = Counter()
    paths = sorted(Path(vis_dir).glob("*_vis.json"))

    for p in paths:
        with open(p, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if "visibility" in data:
                label = data["visibility"]
            elif "label" in data:
                label = data["label"]
            else:
                raise ValueError(f"unknown visibility json format: {p}")
        else:
            label = data

        counter[int(label)] += 1

    return counter, len(paths)


def analyze_rscd_split(txt_or_dir_root, split_name):
    # use existing dataset file layout assumptions from your project
    # try to infer labels from folder names first
    root = Path(txt_or_dir_root)
    counter = Counter()

    # fallback 1: directory per class
    split_dir = root / split_name
    if split_dir.exists():
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if class_dirs:
            name_to_idx = {name: i for i, name in enumerate(RSCD_CLASSES)}
            total = 0
            for d in class_dirs:
                class_name = d.name
                if class_name not in name_to_idx:
                    continue
                idx = name_to_idx[class_name]
                n = len(list(d.glob("*")))
                counter[idx] += n
                total += n
            if total > 0:
                return counter, total

    # fallback 2: use dataset object directly
    from data.rscd_dataset import RSCDRoadCondition
    ds = RSCDRoadCondition(root=str(root), split=split_name)
    total = len(ds)
    for i in range(total):
        _, y = ds[i]
        counter[int(y)] += 1
    return counter, total


def analyze_drivable_split(mask_dir, sample_limit=0):
    paths = sorted(Path(mask_dir).glob("*_drivable_id.png"))
    if sample_limit > 0:
        paths = paths[:sample_limit]

    pixel_counter = Counter()
    image_stats = []

    for p in paths:
        arr = np.array(Image.open(p).convert("L"))
        vals, counts = np.unique(arr, return_counts=True)
        total_pixels = arr.size

        per_image = {int(v): int(c) for v, c in zip(vals, counts)}
        for v, c in per_image.items():
            pixel_counter[v] += c

        bg = per_image.get(0, 0) / total_pixels
        alt = per_image.get(1, 0) / total_pixels
        direct = per_image.get(2, 0) / total_pixels

        image_stats.append({
            "file": p.name,
            "background_ratio": bg,
            "alternative_ratio": alt,
            "direct_ratio": direct,
            "drivable_ratio": alt + direct,
        })

    return pixel_counter, image_stats, len(paths)


def write_drivable_image_stats(save_path, image_stats):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file",
            "background_ratio",
            "alternative_ratio",
            "direct_ratio",
            "drivable_ratio",
        ])
        for row in image_stats:
            writer.writerow([
                row["file"],
                f"{row['background_ratio']:.6f}",
                f"{row['alternative_ratio']:.6f}",
                f"{row['direct_ratio']:.6f}",
                f"{row['drivable_ratio']:.6f}",
            ])


def summarize_drivable_pixels(title, pixel_counter):
    total = sum(pixel_counter.values())
    lines = [f"{title} (total_pixels={total})"]
    for k in sorted(DRIVABLE_IDS.keys()):
        cnt = pixel_counter.get(k, 0)
        ratio = cnt / total if total > 0 else 0.0
        lines.append(f"  {k:>2d} | {DRIVABLE_IDS[k]:<25s} | pixels={cnt:<12d} ratio={ratio:.4f}")
    return "\n".join(lines)


def summarize_drivable_image_stats(title, image_stats):
    if not image_stats:
        return f"{title}\n  no images"

    drivable_ratios = [x["drivable_ratio"] for x in image_stats]
    alt_ratios = [x["alternative_ratio"] for x in image_stats]
    direct_ratios = [x["direct_ratio"] for x in image_stats]

    lines = [title]
    lines.append(f"  num_images={len(image_stats)}")
    lines.append(f"  mean_drivable_ratio={np.mean(drivable_ratios):.4f}")
    lines.append(f"  mean_alternative_ratio={np.mean(alt_ratios):.4f}")
    lines.append(f"  mean_direct_ratio={np.mean(direct_ratios):.4f}")
    lines.append(f"  min_drivable_ratio={np.min(drivable_ratios):.4f}")
    lines.append(f"  max_drivable_ratio={np.max(drivable_ratios):.4f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdd_root", type=str, default="/home/yuhengz3@andrew.cmu.edu/bdd100k")
    parser.add_argument("--rscd_root", type=str, default="/home/yuhengz3@andrew.cmu.edu/rscd/dataset")
    parser.add_argument("--output_dir", type=str, default="analysis/data_distribution")
    parser.add_argument("--drivable_sample_limit", type=int, default=0)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    summary_lines = []

    # bdd time / scene
    for split in ["train", "val", "test"]:
        label_dir = os.path.join(args.bdd_root, "100k_label", "100k", split)
        time_counter, scene_counter, total = analyze_bdd_time_scene_split(label_dir)

        summary_lines.append(summarize_counter(f"BDD {split} time", TIME_CLASSES, time_counter))
        summary_lines.append("")
        summary_lines.append(summarize_counter(f"BDD {split} scene", SCENE_CLASSES, scene_counter))
        summary_lines.append("")

        write_counter_csv(
            os.path.join(args.output_dir, f"bdd_{split}_time.csv"),
            f"bdd_{split}_time",
            TIME_CLASSES,
            time_counter,
        )
        write_counter_csv(
            os.path.join(args.output_dir, f"bdd_{split}_scene.csv"),
            f"bdd_{split}_scene",
            SCENE_CLASSES,
            scene_counter,
        )

    # visibility
    for split in ["train", "val", "test"]:
        vis_dir = os.path.join(args.bdd_root, "visibility_labels", split)
        vis_counter, total = analyze_visibility_split(vis_dir)

        summary_lines.append(summarize_counter(f"BDD {split} visibility", VIS_CLASSES, vis_counter))
        summary_lines.append("")

        write_counter_csv(
            os.path.join(args.output_dir, f"bdd_{split}_visibility.csv"),
            f"bdd_{split}_visibility",
            VIS_CLASSES,
            vis_counter,
        )

    # rscd
    for split in ["train", "test"]:
        road_counter, total = analyze_rscd_split(args.rscd_root, split)

        summary_lines.append(summarize_counter(f"RSCD {split} road_condition", RSCD_CLASSES, road_counter))
        summary_lines.append("")

        write_counter_csv(
            os.path.join(args.output_dir, f"rscd_{split}_road_condition.csv"),
            f"rscd_{split}_road_condition",
            RSCD_CLASSES,
            road_counter,
        )

    # drivable
    drivable_root = os.path.join(args.bdd_root, "bdd100k_drivable_maps", "labels")
    for split in ["train", "val"]:
        mask_dir = os.path.join(drivable_root, split)
        pixel_counter, image_stats, n = analyze_drivable_split(mask_dir, args.drivable_sample_limit)

        summary_lines.append(summarize_drivable_pixels(f"BDD drivable {split} pixel distribution", pixel_counter))
        summary_lines.append("")
        summary_lines.append(summarize_drivable_image_stats(f"BDD drivable {split} image-level ratio summary", image_stats))
        summary_lines.append("")

        write_key_counter_csv(
            os.path.join(args.output_dir, f"bdd_drivable_{split}_pixel_distribution.csv"),
            f"bdd_drivable_{split}_pixel_distribution",
            DRIVABLE_IDS,
            pixel_counter,
        )
        write_drivable_image_stats(
            os.path.join(args.output_dir, f"bdd_drivable_{split}_image_stats.csv"),
            image_stats,
        )

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print("\n".join(summary_lines))
    print(f"\nsaved analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()
