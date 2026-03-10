#!/usr/bin/env python3
"""
Demo script to run YOLO inference on images from the specified dataset split.
Usage: python demo_yolo_output.py [--model {stage2,stage1,official}] [--split {train,val}] [--output OUTPUT_DIR]
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

# -------------------- Argument parsing --------------------
parser = argparse.ArgumentParser(description='Run YOLO inference on dataset images.')
parser.add_argument('--model', type=str, default='stage2',
                    choices=['stage2', 'stage1', 'official'],
                    help='Model to use: stage2 (default), stage1, or official (yolov8m)')
parser.add_argument('--split', type=str, default='val',
                    choices=['train', 'val'],
                    help='Dataset split to visualize: train or val (default: val)')
parser.add_argument('--output', type=str, default='demo_output',
                    help='Output directory for visualized images (default: demo_output)')
args = parser.parse_args()

# -------------------- Determine model path --------------------
if args.model == 'official':
    model_path = 'yolov8m.pt'   # will be auto-downloaded if not present
else:
    # stage1 or stage2: path under runs/detect/yolo/
    base_path = Path('../runs/detect/yolo')
    model_path = base_path / args.model / 'weights' / 'best.pt'
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        exit(1)

print(f"Loading model: {model_path}")
model = YOLO(str(model_path))

# -------------------- Image directory --------------------
# Assume data is under data/yolo/images/{split} relative to current working directory
image_dir = Path('../data/yolo/images') / args.split
if not image_dir.exists():
    print(f"Error: Image directory not found: {image_dir}")
    exit(1)

# Output directory
output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True)

# -------------------- Run inference --------------------
image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))  # adjust extensions as needed
print(f"Found {len(image_files)} images in {image_dir}")

# Process first 10 images (or all if fewer)
for img_path in image_files[:10]:
    print(f"Processing {img_path.name} ...")
    results = model(img_path)
    annotated = results[0].plot()
    out_path = output_dir / f"pred_{img_path.name}"
    cv2.imwrite(str(out_path), annotated)

print(f"Done. Results saved to {output_dir}")