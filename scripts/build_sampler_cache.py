import os
import sys
import json
import argparse
from pathlib import Path

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from data.bdd_dataset import (
    BDDDTimeScene,
    BDDDVisibility,
    get_bdd_root,
    resolve_img_root_for_split,
    resolve_label_root_for_split,
)
from data.rscd_dataset import RSCDRoadCondition
from utils.common import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdd_root", type=str, default="/home/yuhengz3@andrew.cmu.edu/bdd100k")
    parser.add_argument("--rscd_root", type=str, default="/home/yuhengz3@andrew.cmu.edu/rscd/dataset")
    parser.add_argument("--output_dir", type=str, default="cache/sampler_cache")
    parser.add_argument("--road_val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # bdd time/scene
    ts_train = BDDDTimeScene(
        img_root=resolve_img_root_for_split(args.bdd_root, "train"),
        label_dir=resolve_label_root_for_split(args.bdd_root, "train"),
    )

    scene_targets = []
    time_targets = []

    for i in range(len(ts_train)):
        item = ts_train[i]
        scene_targets.append(int(item["labels"]["scene"]))
        time_targets.append(int(item["labels"]["time"]))

    with open(os.path.join(args.output_dir, "bdd_train_scene_targets.json"), "w") as f:
        json.dump(scene_targets, f)

    with open(os.path.join(args.output_dir, "bdd_train_time_targets.json"), "w") as f:
        json.dump(time_targets, f)

    print("saved bdd time/scene targets")

    # bdd visibility
    vis_train = BDDDVisibility(split="train")
    vis_targets = []

    for i in range(len(vis_train)):
        _, y = vis_train[i]
        vis_targets.append(int(y))

    with open(os.path.join(args.output_dir, "bdd_train_visibility_targets.json"), "w") as f:
        json.dump(vis_targets, f)

    print("saved bdd visibility targets")

    # rscd road
    road_train = RSCDRoadCondition(root=args.rscd_root, split="train")
    road_targets = [int(y) for y in road_train.targets]

    with open(os.path.join(args.output_dir, "rscd_train_targets.json"), "w") as f:
        json.dump(road_targets, f)

    print("saved rscd road targets")

    meta = {
        "bdd_root": args.bdd_root,
        "rscd_root": args.rscd_root,
        "seed": args.seed,
        "road_val_ratio": args.road_val_ratio,
        "num_scene": len(scene_targets),
        "num_time": len(time_targets),
        "num_visibility": len(vis_targets),
        "num_road": len(road_targets),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("done. saved cache to:", args.output_dir)


if __name__ == "__main__":
    main()
