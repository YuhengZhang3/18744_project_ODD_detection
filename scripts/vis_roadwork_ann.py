#!/usr/bin/env python3
"""
Compare original ROADWork annotations with augmented (pseudo-labeled) annotations.
For a given image, it generates two images:
  - xxx_orig.jpg: shows original ground truth boxes (green)
  - xxx_aug.jpg: shows augmented boxes (magenta) from the merged labels
Usage: python vis_roadwork_ann.py <image_path> [--output_dir OUTPUT_DIR]
"""

import argparse
import json
import cv2
from pathlib import Path

# Paths to ROADWork annotation JSON files
TRAIN_JSON = Path("../data/roadwork_traj/traj_annotations/trajectories_train_equidistant.json")
VAL_JSON = Path("../data/roadwork_traj/traj_annotations/trajectories_val_equidistant.json")

AUG_LABEL_ROOT = Path("../data/yolo_augmented/labels")  # 包含 train/val 子目录

# Class names for our 6-class mapping (used for display)
CLASS_NAMES = ['pedestrian', 'bicycle', 'vehicle', 'construction_channelizer', 'construction_barrier', 'construction_sign', 'construction_vehicle']


def find_image_in_json(image_name, json_path):
    """Search for image entry in a single JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    for entry in data:
        if entry.get("image", "").endswith(image_name):
            return entry
    return None


def draw_text_with_outline(img, text, pos, font_scale, color, thickness, outline_thickness=6):
    """Draw text with black outline for better visibility."""
    x, y = pos
    # Black outline
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), outline_thickness)
    # Colored text
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def main():
    parser = argparse.ArgumentParser(description="Visualize original vs augmented ROADWork labels.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output images (default: same as input image)")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"Error: Image {img_path} not found.")
        return

    image_name = img_path.name
    stem = img_path.stem

    # Determine which split the image belongs to
    entry = None
    split = None
    if TRAIN_JSON.exists():
        entry = find_image_in_json(image_name, TRAIN_JSON)
        if entry:
            split = 'train'
    if entry is None and VAL_JSON.exists():
        entry = find_image_in_json(image_name, VAL_JSON)
        if entry:
            split = 'val'

    if entry is None:
        print(f"No annotation found for {image_name} in train/val JSONs.")
        return

    # Load original boxes from the JSON entry
    orig_boxes = []  # each element: (x1, y1, x2, y2, category)
    for obj in entry.get("objects", []):
        cat = obj.get("category_id", "unknown")
        bbox = obj.get("bbox")  # [x, y, w, h]
        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            x2 = x + w
            y2 = y + h
            orig_boxes.append((x, y, x2, y2, cat))

    # Read image to get dimensions
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Cannot read image {img_path}.")
        return
    h_img, w_img = img.shape[:2]

    # Load augmented labels if they exist
    aug_label_path = AUG_LABEL_ROOT / split / f"{stem}.txt"
    aug_boxes = []  # each element: (x1, y1, x2, y2, class_id)
    if aug_label_path.exists():
        with open(aug_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, nw, nh = map(float, parts)
                # Convert normalized YOLO format to absolute pixel coordinates
                x1 = int((xc - nw / 2) * w_img)
                y1 = int((yc - nh / 2) * h_img)
                x2 = int((xc + nw / 2) * w_img)
                y2 = int((yc + nh / 2) * h_img)
                # Clip to image boundaries
                x1 = max(0, min(w_img, x1))
                x2 = max(0, min(w_img, x2))
                y1 = max(0, min(h_img, y1))
                y2 = max(0, min(h_img, y2))
                aug_boxes.append((x1, y1, x2, y2, int(cls_id)))
    else:
        print(f"Warning: No augmented labels found at {aug_label_path}")

    # Create two copies of the image
    img_orig = img.copy()
    img_aug = img.copy()

    # --- Draw original boxes (green) on img_orig only ---
    for (x1, y1, x2, y2, cat) in orig_boxes:
        # Ensure coordinates are within image bounds
        x1 = max(0, min(w_img, x1))
        x2 = max(0, min(w_img, x2))
        y1 = max(0, min(h_img, y1))
        y2 = max(0, min(h_img, y2))
        cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0, 255, 0), 4)          # thick green
        # Draw text with outline
        text_pos = (x1, max(y1 - 10, 20))
        draw_text_with_outline(img_orig, cat, text_pos, 2, (0, 255, 0), 4, 6)

    # --- Draw augmented boxes (magenta) on img_aug only ---
    for (x1, y1, x2, y2, cls_id) in aug_boxes:
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
        cv2.rectangle(img_aug, (x1, y1), (x2, y2), (255, 0, 255), 4)          # thick magenta
        # Draw text with outline
        text_pos = (x1, max(y1 - 10, 20))
        draw_text_with_outline(img_aug, label, text_pos, 2, (255, 0, 255), 4, 6)

    # Determine output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = img_path.parent
    out_dir.mkdir(exist_ok=True)

    # Save images
    orig_out_path = out_dir / f"{stem}_orig.jpg"
    aug_out_path = out_dir / f"{stem}_aug.jpg"
    cv2.imwrite(str(orig_out_path), img_orig)
    cv2.imwrite(str(aug_out_path), img_aug)
    print(f"Original visualization saved to {orig_out_path}")
    print(f"Augmented visualization saved to {aug_out_path}")


if __name__ == "__main__":
    main()