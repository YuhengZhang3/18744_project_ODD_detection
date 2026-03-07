import os
import sys
import csv
import math
import argparse
from pathlib import Path

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from models.odd_model import ODDModel
from data.bdd_dataset import TIME_CLASSES, SCENE_CLASSES
from data.rscd_dataset import RSCD_CLASSES
from utils.common import get_device, ensure_dir
from utils.multitask_train import load_model_ckpt


VIS_CLASSES = ["poor", "medium", "good"]

# fixed local paths
BDD_VAL_DIR = "/Users/zhangyuheng/Documents/Study/CMU_Courses/2026Spring/18744/Project/datasets/100k_datasets/val"
DRIVABLE_VAL_DIR = "/Users/zhangyuheng/Documents/Study/CMU_Courses/2026Spring/18744/Project/datasets/bdd100k_drivable_maps/labels/val"


def build_transform():
    return transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict_head(logits, class_names):
    probs = F.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    conf = float(probs[idx].item())
    return class_names[idx], conf


def make_mask_path(image_path):
    stem = Path(image_path).stem
    return Path(DRIVABLE_VAL_DIR) / f"{stem}_drivable_id.png"


def crop_from_drivable_mask(img, mask_path, target_id=2, patch_h_ratio=0.28, patch_w_ratio=0.45):
    import numpy as np

    mask = Image.open(mask_path).convert("L")
    mw, mh = mask.size
    iw, ih = img.size

    if (mw, mh) != (iw, ih):
        mask = mask.resize((iw, ih), resample=Image.NEAREST)

    arr = np.array(mask)
    ys, xs = np.where(arr == target_id)

    # fallback if target id is absent
    if len(xs) == 0 or len(ys) == 0:
        patch_w = int(iw * 0.35)
        patch_h = int(ih * 0.20)
        left = (iw - patch_w) // 2
        top = int(ih * 0.60)
        right = left + patch_w
        bottom = min(top + patch_h, ih)
        return img.crop((left, top, right, bottom)), (left, top, right, bottom)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    bw = x2 - x1 + 1
    bh = y2 - y1 + 1

    patch_w = max(int(bw * patch_w_ratio), int(iw * 0.16))
    patch_h = max(int(bh * patch_h_ratio), int(ih * 0.16))

    cx = int((x1 + x2) / 2)
    cy = int(y2 - 0.38 * bh)

    left = max(cx - patch_w // 2, 0)
    top = max(cy - patch_h // 2, 0)
    right = min(left + patch_w, iw)
    bottom = min(top + patch_h, ih)

    if right - left < patch_w:
        left = max(right - patch_w, 0)
    if bottom - top < patch_h:
        top = max(bottom - patch_h, 0)

    return img.crop((left, top, right, bottom)), (left, top, right, bottom)


def infer_one(model, img, road_img, transform, device):
    full_tensor = transform(img).unsqueeze(0).to(device)
    road_tensor = transform(road_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out_full = model(full_tensor)
        out_road = model(road_tensor)

    pred_time, conf_time = predict_head(out_full["time"], TIME_CLASSES)
    pred_scene, conf_scene = predict_head(out_full["scene"], SCENE_CLASSES)
    pred_vis, conf_vis = predict_head(out_full["visibility"], VIS_CLASSES)
    pred_road, conf_road = predict_head(out_road["road_condition"], RSCD_CLASSES)

    return {
        "time": pred_time,
        "time_conf": conf_time,
        "scene": pred_scene,
        "scene_conf": conf_scene,
        "visibility": pred_vis,
        "visibility_conf": conf_vis,
        "road_condition": pred_road,
        "road_condition_conf": conf_road,
    }


def draw_prediction_overlay(img, box, pred):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)

    draw.rectangle(box, outline=(255, 0, 0), width=4)

    text_lines = [
        f"time: {pred['time']} ({pred['time_conf']:.2f})",
        f"scene: {pred['scene']} ({pred['scene_conf']:.2f})",
        f"visibility: {pred['visibility']} ({pred['visibility_conf']:.2f})",
        f"road: {pred['road_condition']} ({pred['road_condition_conf']:.2f})",
    ]

    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    x0, y0 = 20, 20
    pad = 8
    line_h = 28
    box_w = 430
    box_h = pad * 2 + line_h * len(text_lines)

    draw.rectangle(
        [x0, y0, x0 + box_w, y0 + box_h],
        fill=(0, 0, 0),
        outline=(255, 255, 255),
        width=2
    )

    y = y0 + pad
    for line in text_lines:
        draw.text((x0 + pad, y), line, fill=(255, 255, 255), font=font)
        y += line_h

    return vis


def make_summary_grid(image_paths, vis_dir, save_path, thumb_size=(320, 180), ncols=4, pad=20):
    images = []
    for p in image_paths:
        vis_path = vis_dir / f"{p.stem}_pred.jpg"
        if vis_path.exists():
            img = Image.open(vis_path).convert("RGB")
            img = img.resize(thumb_size)
            images.append((p.name, img))

    if not images:
        return

    n = len(images)
    nrows = math.ceil(n / ncols)
    cell_w, cell_h = thumb_size
    title_h = 26

    canvas_w = ncols * cell_w + (ncols + 1) * pad
    canvas_h = nrows * (cell_h + title_h) + (nrows + 1) * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for idx, (name, img) in enumerate(images):
        r = idx // ncols
        c = idx % ncols

        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + title_h + pad)

        canvas.paste(img, (x, y + title_h))
        draw.text((x, y), name, fill=(0, 0, 0), font=font)

    canvas.save(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="checkpoints_local/stage2_best.pt")
    parser.add_argument("--output_dir", type=str, default="demo_outputs/infer_bdd_val_demo")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--grid_cols", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"

    ensure_dir(str(output_dir))
    ensure_dir(str(vis_dir))

    device = get_device()
    print("device:", device)

    model = ODDModel(freeze_backbone=False).to(device)
    missing, unexpected = load_model_ckpt(model, args.ckpt_path)
    print("loaded ckpt:", args.ckpt_path)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)
    model.eval()

    transform = build_transform()

    image_paths = sorted(Path(BDD_VAL_DIR).glob("*.jpg"))[:args.limit]
    print("num images:", len(image_paths))

    csv_path = output_dir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image",
            "time", "time_conf",
            "scene", "scene_conf",
            "visibility", "visibility_conf",
            "road_condition", "road_condition_conf",
            "visualization_file",
        ])

        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert("RGB")
            mask_path = make_mask_path(img_path)
            road_img, box = crop_from_drivable_mask(img, mask_path)

            pred = infer_one(model, img, road_img, transform, device)

            overlay = draw_prediction_overlay(img, box, pred)
            vis_name = f"{img_path.stem}_pred.jpg"
            vis_save_path = vis_dir / vis_name
            overlay.save(vis_save_path)

            writer.writerow([
                img_path.name,
                pred["time"], f"{pred['time_conf']:.4f}",
                pred["scene"], f"{pred['scene_conf']:.4f}",
                pred["visibility"], f"{pred['visibility_conf']:.4f}",
                pred["road_condition"], f"{pred['road_condition_conf']:.4f}",
                vis_name,
            ])

            print(
                f"[{i+1}/{len(image_paths)}] {img_path.name} | "
                f"time={pred['time']} ({pred['time_conf']:.3f}) | "
                f"scene={pred['scene']} ({pred['scene_conf']:.3f}) | "
                f"vis={pred['visibility']} ({pred['visibility_conf']:.3f}) | "
                f"road={pred['road_condition']} ({pred['road_condition_conf']:.3f})"
            )

    grid_path = output_dir / "summary_grid.jpg"
    make_summary_grid(
        image_paths=image_paths,
        vis_dir=vis_dir,
        save_path=grid_path,
        thumb_size=(320, 180),
        ncols=args.grid_cols,
        pad=20,
    )

    print("saved csv to:", csv_path)
    print("saved visualizations to:", vis_dir)
    print("saved summary grid to:", grid_path)


if __name__ == "__main__":
    main()
