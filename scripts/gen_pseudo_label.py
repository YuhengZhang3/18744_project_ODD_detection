#!/usr/bin/env python3
"""
Script to generate pseudo-labels for missing classes (pedestrian, bicycle, vehicle) on ROADWork images
using a COCO-pretrained YOLO model, then merge them with existing ROADWork labels,
and finally create an augmented dataset in YOLO format.

Steps:
1. Generate pseudo-labels: run inference on all ROADWork images with a pretrained YOLO model,
   filter detections for classes 0,1,2 (pedestrian, bicycle, vehicle) with high confidence,
   and save them in YOLO format.
2. Merge pseudo-labels with original ROADWork labels (already converted to YOLO format).
   For each image, compute IoU between pseudo boxes and original boxes of the same class;
   discard pseudo boxes that overlap significantly (IoU > threshold) with original ones,
   keep the rest.
3. Create a new dataset directory (data/yolo_augmented/) containing:
   - soft links to the original images,
   - merged labels (original + non-overlapping pseudo-labels),
   - a data.yaml file pointing to the new dataset.
"""

import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# -------------------- Configuration --------------------
# Paths (adjust to your project structure)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'
ROADWORK_IMG_DIR = DATA_ROOT / 'roadwork' / 'images'          # all ROADWork images
ORIGINAL_LABEL_DIR = DATA_ROOT / 'yolo' / 'labels'                 # contains train/ and val/ subdirs
OUTPUT_DATASET_DIR = DATA_ROOT / 'yolo_augmented'                  # new dataset root

# Model and inference settings
MODEL_NAME = 'yolov8m.pt'                                           # COCO-pretrained model
CONF_THRESHOLD = 0.7                                                # confidence threshold for pseudo-labels
IOU_THRESHOLD = 0.5                                                 # IoU threshold for discarding overlapping boxes

# Target class mapping (COCO class id -> our class id)
# COCO: person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7, rider? Actually rider is not in COCO? Wait, COCO has person, bicycle, car, motorcycle, bus, truck.
# We only need to map those that are relevant.
COCO_TO_OUR = {
    0: 0,   # person -> pedestrian
    1: 1,   # bicycle -> bicycle
    2: 2,   # car -> vehicle
    3: 1,   # motorcycle -> bicycle (or could be vehicle? but we treat as bicycle for traffic density)
    5: 2,   # bus -> vehicle
    7: 2,   # truck -> vehicle
}

# Splits to process
SPLITS = ['train', 'val']


# -------------------- Helper Functions --------------------
def iou(box1, box2):
    """
    Compute IoU between two boxes in normalized YOLO format [x_center, y_center, width, height].
    """
    # Convert to [x1, y1, x2, y2] format
    x1_c, y1_c, w1, h1 = box1
    x2_c, y2_c, w2, h2 = box2
    x1_min = x1_c - w1 / 2
    y1_min = y1_c - h1 / 2
    x1_max = x1_c + w1 / 2
    y1_max = y1_c + h1 / 2
    x2_min = x2_c - w2 / 2
    y2_min = y2_c - h2 / 2
    x2_max = x2_c + w2 / 2
    y2_max = y2_c + h2 / 2

    # Intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)

    # Union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def load_yolo_labels(label_path):
    """
    Load YOLO label file and return list of (class_id, [x_center, y_center, width, height]).
    """
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                boxes.append((cls, coords))
    return boxes


def save_yolo_labels(label_path, boxes):
    """
    Save list of (class_id, [x_center, y_center, width, height]) to file in YOLO format.
    """
    with open(label_path, 'w') as f:
        for cls, coords in boxes:
            line = f"{cls} " + " ".join(f"{c:.6f}" for c in coords)
            f.write(line + "\n")


# -------------------- Main Script --------------------
def main():
    # Load pretrained YOLO model
    print(f"Loading model {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    # Create output directories
    for split in SPLITS:
        (OUTPUT_DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in SPLITS:
        print(f"\n--- Processing {split} split ---")
        original_label_dir = ORIGINAL_LABEL_DIR / split
        output_label_dir = OUTPUT_DATASET_DIR / 'labels' / split
        output_img_dir = OUTPUT_DATASET_DIR / 'images' / split

        # Get list of all label files (without extension) in original label dir
        label_files = list(original_label_dir.glob("*.txt"))
        if not label_files:
            print(f"Warning: No label files found in {original_label_dir}. Skipping split {split}.")
            continue

        # Process each image
        for label_path in tqdm(label_files, desc=f"Processing {split} images"):
            stem = label_path.stem  # image id without extension
            img_path = ROADWORK_IMG_DIR / f"{stem}.jpg"
            if not img_path.exists():
                raise RuntimeError(f"Warning: Image {img_path} not found, aborting.")
            
            print(f"Generating pseudolabels for {img_path}")

            # ---------- Load original labels ----------
            original_boxes = load_yolo_labels(label_path)  # list of (cls, coords)

            # ---------- Generate pseudo-labels (Task 1) ----------
            # Run inference
            results = model(img_path, verbose=False)[0]
            pseudo_boxes = []
            for det in results.boxes.data:
                x1, y1, x2, y2, conf, cls = det.tolist()
                if conf < CONF_THRESHOLD:
                    continue
                coco_cls = int(cls)
                if coco_cls not in COCO_TO_OUR:
                    continue
                our_cls = COCO_TO_OUR[coco_cls]

                # Convert to YOLO normalized center format
                img_h, img_w = results.orig_shape
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                # Clamp to [0,1] (should already be within)
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                if width * height < 0.0001:  # skip tiny boxes
                    continue
                pseudo_boxes.append((our_cls, [x_center, y_center, width, height], conf))

            # ---------- Merge pseudo-labels with original (Task 2) ----------
            # We will keep all original boxes.
            merged_boxes = original_boxes.copy()  # list of (cls, coords)

            # For each pseudo box, check if it overlaps with any original box of the same class
            for (p_cls, p_coords, conf) in pseudo_boxes:
                # Only consider classes 0,1,2 (our target classes)
                if p_cls not in [0, 1, 2]:
                    continue

                # Find original boxes of the same class
                same_class_orig = [coords for (cls, coords) in original_boxes if cls == p_cls]
                # Compute max IoU with any original box of same class
                max_iou = 0
                for o_coords in same_class_orig:
                    iou_val = iou(p_coords, o_coords)
                    if iou_val > max_iou:
                        max_iou = iou_val
                if max_iou < IOU_THRESHOLD:
                    # No significant overlap, add this pseudo box
                    merged_boxes.append((p_cls, p_coords))

            # ---------- Save merged labels and create image symlink (Task 3) ----------
            # Save merged labels
            out_label_path = output_label_dir / f"{stem}.txt"
            save_yolo_labels(out_label_path, merged_boxes)

            # Create symlink for image (if not already exists)
            out_img_path = output_img_dir / f"{stem}.jpg"
            if not out_img_path.exists():
                try:
                    os.symlink(img_path, out_img_path)
                except:
                    # fallback to copy if symlink fails (e.g., on some systems)
                    shutil.copy2(img_path, out_img_path)

        print(f"Finished {split} split.")

    # ---------- Create data.yaml (Task 3) ----------
    yaml_path = OUTPUT_DATASET_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"""# YOLO dataset configuration for augmented dataset
path: {OUTPUT_DATASET_DIR.resolve()}  # dataset root dir
train: images/train
val: images/val
nc: 6
names: ['pedestrian', 'bicycle', 'vehicle', 'construction_channelizer', 'construction_barrier', 'construction_sign']
""")
    print(f"\nDataset augmented and saved to {OUTPUT_DATASET_DIR}")
    print(f"data.yaml created at {yaml_path}")


if __name__ == '__main__':
    import os
    main()