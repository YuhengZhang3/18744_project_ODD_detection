import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.odd_model import ODDModel
from data.bdd_dataset import TIME_CLASSES, SCENE_CLASSES
from data.rscd_dataset import RSCD_CLASSES


VIS_CLASSES = ["poor", "medium", "good"]
ANOM_CLASSES = ["none", "extreme_weather", "road_blockage_hazard", "road_structure_failure"]


def run_predictions(
    source_directory="../../source_images",
    output_json_directory="output_json",
    checkpoint_path="checkpoints_local/stage2_best.pt",
    drivable_label_directory=None,
    limit=None,
):
    """
    Run unified ODD prediction on all images in source_directory and save one json per image.

    Args:
        source_directory (str): directory containing input images
        output_json_directory (str): directory to save output json files
        checkpoint_path (str): unified model checkpoint
        drivable_label_directory (str | None): optional directory for BDD drivable masks
        limit (int | None): optional max number of images
    """

    # ----------------------------
    # local helper functions
    # ----------------------------
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

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
        probs_list = [round(float(p), 4) for p in probs]
        return class_names[idx], conf, probs_list

    def make_mask_path(image_path):
        if drivable_label_directory is None:
            return None
        stem = Path(image_path).stem
        return Path(drivable_label_directory) / f"{stem}_drivable_id.png"

    def crop_fixed_bottom_center(img):
        w, h = img.size
        patch_w = int(w * 0.35)
        patch_h = int(h * 0.20)
        left = (w - patch_w) // 2
        top = int(h * 0.60)
        right = left + patch_w
        bottom = min(top + patch_h, h)
        return img.crop((left, top, right, bottom)), (left, top, right, bottom)

    def crop_from_drivable_mask(img, mask_path, target_id=2, patch_h_ratio=0.28, patch_w_ratio=0.45):
        import numpy as np

        if mask_path is None or not Path(mask_path).exists():
            return crop_fixed_bottom_center(img)

        mask = Image.open(mask_path).convert("L")
        mw, mh = mask.size
        iw, ih = img.size

        if (mw, mh) != (iw, ih):
            mask = mask.resize((iw, ih), resample=Image.NEAREST)

        arr = np.array(mask)
        ys, xs = np.where(arr == target_id)

        if len(xs) == 0 or len(ys) == 0:
            return crop_fixed_bottom_center(img)

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

    # ----------------------------
    # setup
    # ----------------------------
    source_directory = Path(source_directory)
    output_json_directory = Path(output_json_directory)
    output_json_directory.mkdir(parents=True, exist_ok=True)

    device = get_device()
    transform = build_transform()

    model = ODDModel(freeze_backbone=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("loaded ckpt:", checkpoint_path)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)
    model.eval()

    image_paths = []
    # for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG"]:
    for ext in ["*.JPG"]:
        image_paths.extend(sorted(source_directory.glob(ext)))
    image_paths = sorted(image_paths)

    if limit is not None:
        image_paths = image_paths[:limit]

    print("num images:", len(image_paths))

    # ----------------------------
    # inference loop
    # ----------------------------
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")

        mask_path = make_mask_path(img_path)
        road_img, road_box = crop_from_drivable_mask(img, mask_path)

        full_tensor = transform(img).unsqueeze(0).to(device)
        road_tensor = transform(road_img).unsqueeze(0).to(device)

        with torch.no_grad():
            out_full = model(full_tensor)
            out_road = model(road_tensor)

        pred_time, conf_time, probs_time = predict_head(out_full["time"], TIME_CLASSES)
        pred_scene, conf_scene, probs_scene = predict_head(out_full["scene"], SCENE_CLASSES)
        pred_vis, conf_vis, probs_vis = predict_head(out_full["visibility"], VIS_CLASSES)
        pred_anom, conf_anom, probs_anom = predict_head(out_full["anomalies"], ANOM_CLASSES)
        pred_road, conf_road, probs_road = predict_head(out_road["road_condition"], RSCD_CLASSES)

        result = {
            "image": img_path.name,
            "predictions": {
                "time": {
                    "label": pred_time,
                    "confidence": round(conf_time, 4),
                    "probabilities": probs_time,
                },
                "scene": {
                    "label": pred_scene,
                    "confidence": round(conf_scene, 4),
                    "probabilities": probs_scene,
                },
                "visibility": {
                    "label": pred_vis,
                    "confidence": round(conf_vis, 4),
                    "probabilities": probs_vis,
                },
                "anomalies": {
                    "label": pred_anom,
                    "confidence": round(conf_anom, 4),
                    "probabilities": probs_anom,
                },
                "road_condition": {
                    "label": pred_road,
                    "confidence": round(conf_road, 4),
                    "probabilities": probs_road,
                },
            },
            "road_patch_box": {
                "left": int(road_box[0]),
                "top": int(road_box[1]),
                "right": int(road_box[2]),
                "bottom": int(road_box[3]),
            },
            "drivable_mask_used": str(mask_path) if mask_path is not None and Path(mask_path).exists() else None,
        }

        out_path = output_json_directory / f"{img_path.stem}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(
            f"[{i+1}/{len(image_paths)}] {img_path.name} | "
            f"time={pred_time} ({conf_time:.3f}) | "
            f"scene={pred_scene} ({conf_scene:.3f}) | "
            f"vis={pred_vis} ({conf_vis:.3f}) | "
            f"anom={pred_anom} ({conf_anom:.3f}) | "
            f"road={pred_road} ({conf_road:.3f})"
        )