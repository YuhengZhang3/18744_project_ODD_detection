import os
import sys
import json
import argparse
import subprocess
from collections import defaultdict

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.infer_utils import load_infer_model
from utils.eval_utils import save_json, save_text

from data.bdd_dataset import (
    get_bdd_root,
    resolve_img_root_for_split,
    resolve_label_root_for_split,
    BDDDTimeScene,
    BDDDVisibility,
    BDDDDrivable,
    collate_time_scene,
    TIME_CLASSES,
    SCENE_CLASSES,
    VIS_CLASSES,
    DRIVABLE_CLASSES,
)


def make_simple_loader(dataset, batch_size, num_workers, collate_fn=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )


def update_cls_stats(stats, y_true, y_pred, class_names):
    for t, p in zip(y_true, y_pred):
        t = int(t)
        p = int(p)
        stats[t]["count"] += 1
        stats[t]["correct"] += int(t == p)
        stats[t]["class_name"] = class_names[t]


def summarize_cls_stats(stats, class_names):
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


def format_cls_block(title, overall, rows):
    lines = [title, "overall acc: {:.4f}".format(overall)]
    for r in rows:
        lines.append(
            "  {:>2d} ({:<20s}) acc={:.3f}  {}/{}".format(
                r["class_id"], r["class_name"], r["acc"], r["correct"], r["count"]
            )
        )
    return "\n".join(lines)


def fast_hist(pred, target, num_classes):
    mask = (target >= 0) & (target < num_classes)
    hist = torch.bincount(
        num_classes * target[mask].to(torch.int64) + pred[mask].to(torch.int64),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist


def compute_iou_from_hist(hist):
    ious = []
    for i in range(hist.shape[0]):
        tp = hist[i, i].item()
        fp = hist[:, i].sum().item() - tp
        fn = hist[i, :].sum().item() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        ious.append(iou)
    return ious


def eval_time_scene(model, loader, device):
    stats_time = defaultdict(lambda: {"count": 0, "correct": 0, "class_name": ""})
    stats_scene = defaultdict(lambda: {"count": 0, "correct": 0, "class_name": ""})

    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device, non_blocking=True)
            y_time = batch["labels"]["time"].to(device, non_blocking=True)
            y_scene = batch["labels"]["scene"].to(device, non_blocking=True)

            out = model(imgs)
            pred_time = out["time"].argmax(dim=1)
            pred_scene = out["scene"].argmax(dim=1)

            update_cls_stats(
                stats_time,
                y_time.detach().cpu().tolist(),
                pred_time.detach().cpu().tolist(),
                TIME_CLASSES,
            )
            update_cls_stats(
                stats_scene,
                y_scene.detach().cpu().tolist(),
                pred_scene.detach().cpu().tolist(),
                SCENE_CLASSES,
            )

    time_overall, time_rows = summarize_cls_stats(stats_time, TIME_CLASSES)
    scene_overall, scene_rows = summarize_cls_stats(stats_scene, SCENE_CLASSES)

    return {
        "time": {"overall": time_overall, "per_class": time_rows},
        "scene": {"overall": scene_overall, "per_class": scene_rows},
    }


def eval_visibility(model, loader, device):
    stats_vis = defaultdict(lambda: {"count": 0, "correct": 0, "class_name": ""})

    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(imgs)
            pred = out["visibility"].argmax(dim=1)

            update_cls_stats(
                stats_vis,
                labels.detach().cpu().tolist(),
                pred.detach().cpu().tolist(),
                VIS_CLASSES,
            )

    vis_overall, vis_rows = summarize_cls_stats(stats_vis, VIS_CLASSES)
    return {
        "visibility": {"overall": vis_overall, "per_class": vis_rows},
    }


def _extract_drivable_batch(batch):
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("tuple/list batch has fewer than 2 elements")

    if isinstance(batch, dict):
        imgs = None
        if "images" in batch:
            imgs = batch["images"]
        elif "image" in batch:
            imgs = batch["image"]

        masks = None
        if "labels" in batch and isinstance(batch["labels"], dict) and "drivable" in batch["labels"]:
            masks = batch["labels"]["drivable"]

        if imgs is None or masks is None:
            raise ValueError(
                "dict batch missing image or labels['drivable']; got keys={}".format(list(batch.keys()))
            )
        return imgs, masks

    raise ValueError("unsupported drivable batch type: {}".format(type(batch)))


def eval_drivable(model, loader, device):
    hist = torch.zeros((3, 3), dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs, masks = _extract_drivable_batch(batch)

            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            out = model(imgs)
            pred = out["drivable"].argmax(dim=1)

            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            if masks.ndim == 4 and masks.shape[1] > 1:
                masks = masks.argmax(dim=1)

            if pred.shape[-2:] != masks.shape[-2:]:
                pred = F.interpolate(
                    pred.unsqueeze(1).float(),
                    size=masks.shape[-2:],
                    mode="nearest",
                ).squeeze(1).long()

            hist += fast_hist(pred.reshape(-1), masks.reshape(-1), 3).cpu()

    ious = compute_iou_from_hist(hist)
    miou_all = sum(ious) / len(ious)
    fg_ious = ious[1:]
    miou_fg = sum(fg_ious) / len(fg_ious)

    return {
        "drivable": {
            "iou_per_class": {
                DRIVABLE_CLASSES[0]: ious[0],
                DRIVABLE_CLASSES[1]: ious[1],
                DRIVABLE_CLASSES[2]: ious[2],
            },
            "miou_all": miou_all,
            "miou_fg": miou_fg,
            "foreground_iou": ious[2],
        }
    }


def run_road_script(script_path, ckpt_path, output_dir):
    cmd = [
        sys.executable, script_path,
        "--ckpt_path", ckpt_path,
        "--output_dir", output_dir,
    ]
    subprocess.run(cmd, check=True)


def write_summary_md(output_dir, ckpt_path, bdd_result, road_aux_json):
    lines = []
    lines.append("# Unified Evaluation Summary")
    lines.append("")
    lines.append("checkpoint: `{}`".format(ckpt_path))
    lines.append("")

    lines.append("## BDD")
    lines.append("")
    lines.append("- time acc: {:.4f}".format(bdd_result["time"]["overall"]))
    lines.append("- scene acc: {:.4f}".format(bdd_result["scene"]["overall"]))
    lines.append("- visibility acc: {:.4f}".format(bdd_result["visibility"]["overall"]))
    lines.append("- drivable mIoU all: {:.4f}".format(bdd_result["drivable"]["miou_all"]))
    lines.append("- drivable mIoU fg: {:.4f}".format(bdd_result["drivable"]["miou_fg"]))
    lines.append("- drivable foreground IoU: {:.4f}".format(bdd_result["drivable"]["foreground_iou"]))
    lines.append("")

    if road_aux_json is not None:
        lines.append("## Road")
        lines.append("")
        try:
            direct = road_aux_json["test"]["direct"]["road_condition_overall"]
            infer_style = road_aux_json["test"]["infer_style"]["infer_style_road_condition_overall"]
            lines.append("- road direct acc: {:.4f}".format(direct))
            lines.append("- road infer-style acc: {:.4f}".format(infer_style))
        except Exception:
            lines.append("- road results: see road_aux/road_aux_eval.json")
        lines.append("")
        lines.append("- road aux file: `road_aux/road_aux_eval.json`")
        lines.append("- road relaxed file: `road_relaxed/road_relaxed.txt`")

    save_text("\n".join(lines), os.path.join(output_dir, "summary.md"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--bdd_root", type=str, default="")
    parser.add_argument("--bdd_split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="eval_outputs/eval_all_heads")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--skip_road", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "bdd"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "road_aux"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "road_relaxed"), exist_ok=True)

    model, device, load_meta = load_infer_model(ckpt_path=args.ckpt_path)

    bdd_root = args.bdd_root if args.bdd_root else get_bdd_root()

    # time + scene
    img_root = resolve_img_root_for_split(bdd_root, args.bdd_split)
    label_root = resolve_label_root_for_split(bdd_root, args.bdd_split)
    ds_time_scene = BDDDTimeScene(img_root=img_root, label_dir=label_root)
    loader_time_scene = make_simple_loader(
        ds_time_scene,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_time_scene,
    )
    result_time_scene = eval_time_scene(model, loader_time_scene, device)

    # visibility
    ds_visibility = BDDDVisibility(split=args.bdd_split)
    loader_visibility = make_simple_loader(
        ds_visibility,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=None,
    )
    result_visibility = eval_visibility(model, loader_visibility, device)

    # drivable
    drv_split = "val" if args.bdd_split == "test" else args.bdd_split
    ds_drivable = BDDDDrivable(split=drv_split)
    loader_drivable = make_simple_loader(
        ds_drivable,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=None,
    )
    result_drivable = eval_drivable(model, loader_drivable, device)

    bdd_result = {}
    bdd_result.update(result_time_scene)
    bdd_result.update(result_visibility)
    bdd_result.update(result_drivable)

    save_json(
        {
            "ckpt_path": args.ckpt_path,
            "device": str(device),
            "bdd_root": bdd_root,
            "bdd_split": args.bdd_split,
            "drivable_split": drv_split,
            "load_meta": load_meta,
            "bdd_result": bdd_result,
        },
        os.path.join(args.output_dir, "bdd", "bdd_eval.json"),
    )

    lines = []
    lines.append(format_cls_block(
        "BDD time",
        bdd_result["time"]["overall"],
        bdd_result["time"]["per_class"],
    ))
    lines.append("")
    lines.append(format_cls_block(
        "BDD scene",
        bdd_result["scene"]["overall"],
        bdd_result["scene"]["per_class"],
    ))
    lines.append("")
    lines.append(format_cls_block(
        "BDD visibility",
        bdd_result["visibility"]["overall"],
        bdd_result["visibility"]["per_class"],
    ))
    lines.append("")
    lines.append("BDD drivable")
    lines.append("mIoU all: {:.4f}".format(bdd_result["drivable"]["miou_all"]))
    lines.append("mIoU fg: {:.4f}".format(bdd_result["drivable"]["miou_fg"]))
    lines.append("foreground IoU: {:.4f}".format(bdd_result["drivable"]["foreground_iou"]))
    lines.append("per class:")
    for k, v in bdd_result["drivable"]["iou_per_class"].items():
        lines.append("  {}: {:.4f}".format(k, v))
    save_text("\n".join(lines), os.path.join(args.output_dir, "bdd", "bdd_eval.txt"))

    road_aux_json = None
    if not args.skip_road:
        run_road_script(
            script_path=os.path.join(root_dir, "scripts", "eval_road_aux_heads.py"),
            ckpt_path=args.ckpt_path,
            output_dir=os.path.join(args.output_dir, "road_aux"),
        )
        run_road_script(
            script_path=os.path.join(root_dir, "scripts", "eval_road_relaxed.py"),
            ckpt_path=args.ckpt_path,
            output_dir=os.path.join(args.output_dir, "road_relaxed"),
        )

        road_aux_json_path = os.path.join(args.output_dir, "road_aux", "road_aux_eval.json")
        if os.path.exists(road_aux_json_path):
            with open(road_aux_json_path, "r") as f:
                road_aux_json = json.load(f)

    write_summary_md(
        output_dir=args.output_dir,
        ckpt_path=args.ckpt_path,
        bdd_result=bdd_result,
        road_aux_json=road_aux_json,
    )

    print("saved all evaluation to:", args.output_dir)
    print("BDD json:", os.path.join(args.output_dir, "bdd", "bdd_eval.json"))
    print("BDD txt:", os.path.join(args.output_dir, "bdd", "bdd_eval.txt"))
    if not args.skip_road:
        print("Road aux dir:", os.path.join(args.output_dir, "road_aux"))
        print("Road relaxed dir:", os.path.join(args.output_dir, "road_relaxed"))
    print("Summary:", os.path.join(args.output_dir, "summary.md"))


if __name__ == "__main__":
    main()
