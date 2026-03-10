#!/usr/bin/env python3
"""
Visualize original ROADWork annotations for a given image.
Usage: python visualize_roadwork_ann.py <image_path>
Example: python vis_roadwork_ann.py data/roadwork/images/new_york_city_1932e9f...jpg
"""

import sys
import json
import cv2
from pathlib import Path

# Paths to ROADWork annotation JSON files (adjust if needed)
TRAIN_JSON = Path("data/roadwork/traj_annotations/trajectories_train_equidistant.json")
VAL_JSON = Path("data/roadwork/traj_annotations/trajectories_val_equidistant.json")


def find_image_in_json(image_name, json_path):
    """Search for image entry in a single JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    for entry in data:
        # entry["image"] might be like "images/xxx.jpg"
        if entry.get("image", "").endswith(image_name):
            return entry
    return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_roadwork_ann.py <image_path>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Error: Image {img_path} not found.")
        sys.exit(1)

    # Extract just the filename (e.g., "xxx.jpg")
    image_name = img_path.name

    # Try to find the annotation in train or val JSON
    annotation = None
    if TRAIN_JSON.exists():
        annotation = find_image_in_json(image_name, TRAIN_JSON)
    if annotation is None and VAL_JSON.exists():
        annotation = find_image_in_json(image_name, VAL_JSON)

    if annotation is None:
        print(f"No annotation found for {image_name} in train/val JSONs.")
        sys.exit(1)

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Cannot read image {img_path}.")
        sys.exit(1)

    h, w = img.shape[:2]

    # Draw each object
    for obj in annotation.get("objects", []):
        category = obj.get("category_id", "unknown")
        bbox = obj.get("bbox")  # [x, y, width, height]
        if not bbox or len(bbox) != 4:
            continue
        x, y, bw, bh = bbox
        # Convert to integers for drawing
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + bw)
        y2 = int(y + bh)
        # Clip to image boundaries (just in case)
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))

        # Choose a color based on category (optional)
        color = (0, 255, 0)  # default green
        if "work" in category.lower():
            color = (255, 0, 0)  # blue for work equipment
        elif "vehicle" in category.lower():
            color = (0, 0, 255)  # red for vehicles

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, category, (x1, max(y1-5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save output image
    output_path = img_path.with_stem(img_path.stem + "_viz").with_suffix(".jpg")
    cv2.imwrite(str(output_path), img)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()