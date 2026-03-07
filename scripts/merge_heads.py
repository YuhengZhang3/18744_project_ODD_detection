import os
import sys
import argparse
import torch

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from models.odd_model import ODDModel


def load_state(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    return ckpt.get("model", ckpt), ckpt


def copy_prefix(src_sd, dst_sd, prefix):
    copied = []
    for k, v in src_sd.items():
        if k.startswith(prefix):
            dst_sd[k] = v
            copied.append(k)
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time_scene_ckpt",
        type=str,
        default="checkpoints_time_scene/best.pt",
    )
    parser.add_argument(
        "--visibility_ckpt",
        type=str,
        default="checkpoints_visibility/best.pt",
    )
    parser.add_argument(
        "--road_ckpt",
        type=str,
        default="checkpoints_road_condition_rscd/best.pt",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="checkpoints_joint/merged_init.pt",
    )
    args = parser.parse_args()

    # build a fresh model just to verify keys are valid
    model = ODDModel(freeze_backbone=False)
    fresh_sd = model.state_dict()

    ts_sd, ts_raw = load_state(args.time_scene_ckpt)
    vis_sd, vis_raw = load_state(args.visibility_ckpt)
    road_sd, road_raw = load_state(args.road_ckpt)

    merged_sd = dict(fresh_sd)

    # use time_scene as base
    for k, v in ts_sd.items():
        if k in merged_sd and merged_sd[k].shape == v.shape:
            merged_sd[k] = v

    # overwrite only selected heads
    vis_keys = copy_prefix(vis_sd, merged_sd, "heads.visibility.")
    road_keys = copy_prefix(road_sd, merged_sd, "heads.road_condition.")

    missing = []
    unexpected = []
    model.load_state_dict(merged_sd, strict=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    out_ckpt = {
        "model": merged_sd,
        "source_time_scene": args.time_scene_ckpt,
        "source_visibility": args.visibility_ckpt,
        "source_road_condition": args.road_ckpt,
        "merge_rule": "time_scene as base, overwrite heads.visibility and heads.road_condition",
        "copied_visibility_keys": vis_keys,
        "copied_road_condition_keys": road_keys,
    }

    torch.save(out_ckpt, args.out)

    print("saved merged checkpoint to:", args.out)
    print("time_scene base:", args.time_scene_ckpt)
    print("visibility overwrite:", args.visibility_ckpt)
    print("road_condition overwrite:", args.road_ckpt)
    print("num visibility keys copied:", len(vis_keys))
    print("num road_condition keys copied:", len(road_keys))


if __name__ == "__main__":
    main()
