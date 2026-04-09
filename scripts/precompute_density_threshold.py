#!/usr/bin/env python3
"""
Script to compute calibration thresholds (max_count and max_area_ratio) for traffic density calculation.
Uses the trained YOLO model on the validation set and saves the 95th percentiles to a JSON file.
Usage: python scripts/compute_density_stats.py --model path/to/best.pt --data data.yaml --output thresholds.json
"""

import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import yaml
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Compute density calibration thresholds.')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained YOLO weights (e.g., runs/detect/yolo/stage2/weights/best.pt)')
    parser.add_argument('--data', type=str, default='../data/yolo/data.yaml',
                        help='Path to data.yaml file (e.g., data/yolo/data.yaml)')
    parser.add_argument('--output', type=str, default='density_thresholds.json',
                        help='Output JSON file path')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--percentile', type=float, default=95,
                        help='Percentile to use for thresholds (default: 95)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)

    # Load data.yaml to get validation image path
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    # data.yaml may contain path, images/val etc. Need to construct absolute path to val images
    # Assume data.yaml has 'val' field pointing to relative or absolute directory.
    val_dir = Path(data_config['val'])
    if not val_dir.is_absolute():
        # If relative, resolve relative to the location of data.yaml or current dir.
        # We'll assume relative to data.yaml's directory.
        base = Path(args.data).parent
        val_dir = base / val_dir
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    # Get all image files (common extensions)
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(val_dir.glob(f'*{ext}'))
    print(f"Found {len(image_paths)} images in {val_dir}")

    # Hardcoded class IDs based on our trained model (0: pedestrian, 1: bicycle, 2: vehicle)
    CLASS_IDS = {
        'pedestrian': 0,
        'bicycle': 1,
        'vehicle': 2
    }

    # Data storage for each category
    data = {
        'car': {'counts': [], 'area_ratios': []},
        'pedestrian': {'counts': [], 'area_ratios': []},
        'bicycle': {'counts': [], 'area_ratios': []}
    }

    # Iterate over validation images
    for img_path in tqdm(image_paths, desc="Processing validation images"):
        results = model(img_path, conf=args.conf, verbose=False)[0]
        img_h, img_w = results.orig_shape
        img_area = img_w * img_h

        # Initialize per-image counts and area sums for our categories
        per_img_counts = {'car': 0, 'pedestrian': 0, 'bicycle': 0}
        per_img_areas = {'car': 0.0, 'pedestrian': 0.0, 'bicycle': 0.0}

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0].cpu())
                # Determine category based on hardcoded IDs
                if cls_id == CLASS_IDS['pedestrian']:
                    cat = 'pedestrian'
                elif cls_id == CLASS_IDS['bicycle']:
                    cat = 'bicycle'
                elif cls_id == CLASS_IDS['vehicle']:
                    cat = 'car'
                else:
                    continue   # ignore work zone classes
                per_img_counts[cat] += 1
                # Calculate box area
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                area = (x2 - x1) * (y2 - y1)
                per_img_areas[cat] += area

        # Store per image statistics
        for cat in ['car', 'pedestrian', 'bicycle']:
            data[cat]['counts'].append(per_img_counts[cat])
            data[cat]['area_ratios'].append(per_img_areas[cat] / img_area if img_area > 0 else 0.0)

    # Need some hard-coded lower bound, because in BDD and ROADwork we don't have perfect data showing "full bike traffic" or "full pedestrian traffic"
    # This acts as the "common sense for a traffic jam"
    MIN_FLOORS = {
            'car': (15.0, 0.2),
            'pedestrian': (10.0, 0.1),
            'bicycle': (10.0, 0.1)
    }
    # Compute thresholds (95th percentile by default)
    thresholds = {}
    for cat in ['car', 'pedestrian', 'bicycle']:
        counts = np.array(data[cat]['counts'])
        area_ratios = np.array(data[cat]['area_ratios'])
        
        min_cnt_floor, min_area_floor = MIN_FLOORS[cat]
        
        if len(counts) > 0:
            calc_max_count = np.percentile(counts, args.percentile)
            calc_max_area = np.percentile(area_ratios, args.percentile)
            
            # maybe the max count in dataset is less than our common sense for a traffic jam?
            # use common sense instead
            thresholds[cat] = {
                'max_count': float(max(calc_max_count, min_cnt_floor)),
                'max_area_ratio': float(max(calc_max_area, min_area_floor))
            }
        else:
            thresholds[cat] = {'max_count': float(min_cnt_floor), 'max_area_ratio': float(min_area_floor)}

    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"Thresholds saved to {output_path}")
    print("Values:")
    for cat, vals in thresholds.items():
        print(f"  {cat}: max_count={vals['max_count']:.2f}, max_area_ratio={vals['max_area_ratio']:.3f}")

if __name__ == '__main__':
    main()