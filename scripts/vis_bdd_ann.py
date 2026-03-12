#!/usr/bin/env python3
"""
Visualize BDD100K annotations for a given image.
Usage: python visualize_bdd_ann.py <image_path>

Example: python visualize_bdd_ann.py data/bdd100k_val/images/b6d0b9d1-bfbc7861.jpg
"""

import sys
import json
import cv2
from pathlib import Path

# Default label directory relative to image path
# Assumes images are in .../images/ and labels in .../labels/
def find_label_path(image_path):
    img_path = Path(image_path)
    # Try to replace 'images' with 'labels' in the path
    parts = list(img_path.parts)
    if 'images' in parts:
        idx = parts.index('images')
        parts[idx] = 'labels'
        label_path = Path(*parts).with_suffix('.json')
        if label_path.exists():
            return label_path
    # Fallback: look in a sibling 'labels' directory
    label_dir = img_path.parent.parent / 'labels'
    label_path = label_dir / img_path.stem
    return label_path.with_suffix('.json')

def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_bdd_ann.py <image_path>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Error: Image not found: {img_path}")
        sys.exit(1)

    # Locate annotation file
    label_path = find_label_path(img_path)
    if not label_path.exists():
        print(f"Error: Annotation not found: {label_path}")
        sys.exit(1)

    # Load JSON
    with open(label_path, 'r') as f:
        data = json.load(f)

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Cannot read image {img_path}")
        sys.exit(1)

    # Get image dimensions (for sanity)
    h, w = img.shape[:2]

    # Process each frame (usually only one)
    for frame in data.get('frames', []):
        for obj in frame.get('objects', []):
            category = obj.get('category', 'unknown')
            box = obj.get('box2d')
            if not box:
                continue
            x1 = int(box.get('x1', 0))
            y1 = int(box.get('y1', 0))
            x2 = int(box.get('x2', 0))
            y2 = int(box.get('y2', 0))
            # Clip to image boundaries
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))

            # Choose color based on category
            if category in ['motor', 'rider']:
                color = (0, 0, 255)  # red for motor/rider
            elif category == 'car':
                color = (0, 255, 0)  # green
            else:
                color = (255, 0, 0)  # blue for others

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, category, (x1, max(y1-5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save output
    output_path = img_path.with_stem(img_path.stem + '_viz').with_suffix('.jpg')
    cv2.imwrite(str(output_path), img)
    print(f"Visualization saved to {output_path}")

if __name__ == '__main__':
    main()