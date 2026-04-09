#!/usr/bin/env python3
"""
Demo script to run YOLO inference on images from the specified dataset split.
Usage: python demo_yolo_output.py [--model {stage2,stage1,official}] [--split {train,val}] [--output OUTPUT_DIR]
"""

import argparse
from pathlib import Path
import sys
import cv2
from ultralytics import YOLO
import random # Import the random module

# Add project root to sys.path for absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.traffic_workzone_analyzer import TrafficWorkzoneAnalyzer
from visualization_utils import draw_info_panel # Import draw_info_panel

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
parser.add_argument('--seed', type=int, default=None,
                    help='Random seed for shuffling image selection (default: None for no shuffling)')
args = parser.parse_args()

# -------------------- Determine model path --------------------
if args.model == 'official':
    model_path = 'yolov8m.pt'   # will be auto-downloaded if not present
else:
    # stage1 or stage2: path under runs/detect/yolo/
    base_path = Path('../runs/detect/yolo11')
    model_path = base_path / args.model / 'weights' / 'best.pt'
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        exit(1)

print(f"Loading YOLO model for inference: {model_path}")
model = YOLO(str(model_path))

# -------------------- Initialize TrafficWorkzoneAnalyzer --------------------
analyzer_model_path = model_path # Use the same YOLO model for the analyzer
thresholds_path = Path(__file__).parent / "density_thresholds.json"
if not thresholds_path.exists():
    print(f"Error: Thresholds file not found at {thresholds_path}")
    exit(1)

print(f"Initializing TrafficWorkzoneAnalyzer with model: {analyzer_model_path} and thresholds: {thresholds_path}")
traffic_analyzer = TrafficWorkzoneAnalyzer(
    model_path=str(analyzer_model_path),
    device='cpu', # using cpu for analyzer to avoid double cuda context issues with YOLO or if only CPU is available
    thresholds_path=str(thresholds_path)
)


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

# Apply seed if provided and shuffle
if args.seed is not None:
    random.seed(args.seed)
    random.shuffle(image_files)
    print(f"Shuffled image selection with seed: {args.seed}")

# Process first 10 images (or all if fewer)
for img_path in image_files[:10]:
    print(f"Processing {img_path.name} ...")
    
    # Run YOLO inference
    results = model(img_path)
    annotated = results[0].plot() # This is a numpy array (image)
    
    # Run traffic analysis
    analysis_result = traffic_analyzer.analyze_image(str(img_path))
    
    # Draw analysis info on the image
    draw_info_panel(annotated, analysis_result['traffic_density'], analysis_result['work_zone'])

    out_path = output_dir / f"pred_{img_path.name}"
    cv2.imwrite(str(out_path), annotated)

print(f"Done. Results saved to {output_dir}")