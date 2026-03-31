#!/usr/bin/env python3
"""
Convert ROADWork and BDD100K datasets to YOLO format for training.

Original label formats:
- ROADWork: Two large JSON files (train/val) where each entry corresponds to one image.
  Each entry contains "image" (path) and "objects" list with "category_id" and "bbox" [x,y,w,h].
- BDD100K: Each image has its own JSON file (as shown in the example). The JSON contains a "frames" list,
  where each frame has an "objects" list. Objects have "category" (string) and "box2d" {x1,y1,x2,y2}.

Target YOLO format:
- For each image, a .txt file with the same base name.
- Each line: <class_id> <x_center> <y_center> <width> <height> (all normalized to [0,1]).
- Class mapping (6 classes):
    0: pedestrian (COCO person, ROADWork Worker, Police Officer)
    1: bicycle (COCO bicycle, motorcycle)
    2: vehicle (COCO car, truck, bus, Police Vehicle)
    3: construction_channelizer (Cone, Drum, Tubular Marker, Vertical Panel)
    4: construction_barrier (Barrier, Barricade, Fence, Work Equipment)
    5: construction_sign (all Temporary Traffic Control Sign variants, Arrow Board, Message Board)
    6: construction_vehicle (ROADwork Work Vehicle)
"""

import json
import shutil
from pathlib import Path
from abc import ABC, abstractmethod
import cv2
import os

class BaseConverter(ABC):
    """Base converter for any dataset to YOLO format."""

    def __init__(self, src_root, dst_root, split, image_ext='.jpg', use_symlink=True):
        """
        Args:
            src_root (str or Path): Root directory of the original dataset.
            dst_root (str or Path): Root directory for YOLO-formatted data.
            split (str): 'train' or 'val'.
            image_ext (str): Extension of image files.
            use_symlink (bool): If True, create symbolic links for images instead of copying.
        """
        self.src_root = Path(src_root)
        self.dst_root = Path(dst_root)
        self.split = split
        self.image_ext = image_ext
        self.use_symlink = use_symlink

        # Create output directories
        self.img_out_dir = self.dst_root / 'images' / split
        self.label_out_dir = self.dst_root / 'labels' / split
        self.img_out_dir.mkdir(parents=True, exist_ok=True)
        self.label_out_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def parse_annotations(self):
        """
        Yield dictionaries containing:
            - image_id (str): unique identifier (without extension) to name output files.
            - image_src_path (Path): full path to the source image.
            - objects (list): list of tuples (original_category, bbox) where bbox is [x, y, w, h]
                              in absolute pixel coordinates (x,y top-left).
        """
        pass

    @abstractmethod
    def map_category(self, original_category):
        """
        Map the original category string to target class ID (0-5) or None if ignored.
        """
        pass

    def convert(self):
        """Main conversion loop."""
        for item in self.parse_annotations():
            image_id = item['image_id']
            src_img_path = item['image_src_path']
            objects = item['objects']

            # Read image to get dimensions
            img = cv2.imread(str(src_img_path))
            if img is None:
                raise RuntimeError(f"Cannot read image {src_img_path}. Aborting.")
            h, w = img.shape[:2]

            # Prepare YOLO lines
            yolo_lines = []
            for cat, bbox in objects:
                target_id = self.map_category(cat)
                if target_id is None:
                    continue

                x, y, bw, bh = bbox
                # Convert to normalized center format
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h

                # Clamp to [0,1] and ignore tiny boxes
                if nw * nh < 0.0001:
                    continue
                yolo_lines.append(f"{target_id} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")

            # Write label file (always create, even if empty)
            label_path = self.label_out_dir / f"{image_id}.txt"
            if yolo_lines:
                with open(label_path, 'w') as f:
                    f.write("\n".join(yolo_lines))
            else:
                # Create empty label file and notify
                with open(label_path, 'w') as f:
                    pass
                print(f"Image {image_id} has no valid objects, empty label file created.")

            # Handle image (copy or symlink)
            dst_img_path = self.img_out_dir / f"{image_id}{self.image_ext}"
            if self.use_symlink:
                if not dst_img_path.exists():
                    os.symlink(src_img_path, dst_img_path)
            else:
                shutil.copy2(src_img_path, dst_img_path)

class RoadWorkConverter(BaseConverter):
    """
    Converter for ROADWork dataset (traj_annotations JSON files).
    Expects JSON format as described: list of entries, each with "image" and "objects".
    """

    # Mapping from original category strings to target IDs.
    CATEGORY_MAP = {
        # pedestrians
        "Worker": 0,
        "Police Officer": 0,
        # bicycles (none in ROADWork, but keep for completeness)
        # vehicles
        "Work Vehicle": 6,          # NOTE: work vehicle is now a class on its own
        "Police Vehicle": 2,        # this still belongs to normal vehicles
        # construction channelizers
        "Cone": 3,
        "Drum": 3,
        "Tubular Marker": 3,
        "Vertical Panel": 3,
        # construction barriers
        "Barrier": 4,
        "Barricade": 4,
        "Fence": 4,
        "Work Equipment": 4,
        # construction signs
        "Temporary Traffic Control Sign": 5,
        "Arrow Board": 5,
        "Temporary Traffic Control Message Board": 5,
    }

    # Also include all subcategories of Temporary Traffic Control Sign by prefix match
    # We'll handle them in map_category.

    def __init__(self, json_path, src_root, dst_root, split, **kwargs):
        """
        Args:
            json_path (str or Path): Path to the ROADWork JSON file (e.g., trajectories_train_equidistant.json).
        """
        super().__init__(src_root, dst_root, split, **kwargs)
        self.json_path = Path(json_path)
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)  # Should be a list

    def parse_annotations(self):
        for entry in self.data:
            # Extract image filename from "image" field, e.g., "images/xxxx.jpg"
            image_rel = entry.get("image")
            if not image_rel:
                continue
            image_name = Path(image_rel).name
            image_id = image_name.rsplit('.', 1)[0]  # remove extension

            # Construct source image path: src_root / image_rel (e.g., data/roadwork/images/xxxx.jpg)
            src_img_path = self.src_root / image_rel

            objects = []
            for obj in entry.get("objects", []):
                cat = obj.get("category_id")
                bbox = obj.get("bbox")  # [x, y, w, h]
                if cat and bbox and len(bbox) == 4:
                    objects.append((cat, bbox))

            yield {
                "image_id": image_id,
                "image_src_path": src_img_path,
                "objects": objects
            }

    def map_category(self, original_category):
        # Direct lookup
        if original_category in self.CATEGORY_MAP:
            return self.CATEGORY_MAP[original_category]

        # NOTE: these are several special sub-classes inside roadwork dataset
        # They should belong to class 5 just like normal "Temporary Traffic Control Sign"
        if original_category.startswith("Temporary Traffic Control Sign:"):
            return 5  # construction_sign

        # All other categories (e.g., "Bike Lane", "Other Roadwork Objects") are ignored
        return None


class BDDConverter(BaseConverter):
    """
    Converter for BDD100K dataset where each image has its own JSON file.
    The JSON structure contains a "frames" list; we assume the first frame contains the objects.
    Each object has "category" and "box2d" with x1,y1,x2,y2.
    """

    # Mapping from BDD categories to target IDs.
    # Based on discussion: person, rider -> 0; bicycle, motorcycle -> 1; car, truck, bus -> 2.
    CATEGORY_MAP = {
        "person": 0,
        "rider": 0, # need double check, should rider be considered? 
        "bike": 1,
        "motor": 1,
        "car": 2,
        "truck": 2,
        "bus": 2,
    }

    def __init__(self, src_root, dst_root, split, json_dir=None, json_files=None, **kwargs):
        """
        Args:
            src_root: root dir of bdd_100k_val
            dst_root: YOLO output root dir
            split: 'train' 或 'val'
            json_dir: dir containing all json files (NOTE: only providing one of this and below is OK)
            json_files: list of the json files to use (NOTE: only providing one of this and above is OK)
        """
        super().__init__(src_root, dst_root, split, **kwargs)
        if json_files is not None:
            self.json_files = json_files
        elif json_dir is not None:
            self.json_dir = Path(json_dir)
            self.json_files = list(self.json_dir.glob("*.json"))
        else:
            raise ValueError("Either json_dir or json_files must be provided")

    def parse_annotations(self):
        for json_path in self.json_files:
            with open(json_path, 'r') as f:
                data = json.load(f)

            image_id = json_path.stem  # filename without extension

            # Construct image path: src_root / "images" / f"{image_id}.jpg"
            src_img_path = self.src_root / "images" / f"{image_id}.jpg"

            # Extract objects from the first frame
            frames = data.get("frames", [])
            if not frames:
                continue
            frame = frames[0]
            objects = []
            for obj in frame.get("objects", []):
                cat = obj.get("category")
                box = obj.get("box2d")
                if cat and box:
                    x1 = box.get("x1", 0)
                    y1 = box.get("y1", 0)
                    x2 = box.get("x2", 0)
                    y2 = box.get("y2", 0)
                    w = x2 - x1
                    h = y2 - y1
                    if w > 0 and h > 0:
                        objects.append((cat, [x1, y1, w, h]))

            yield {
                "image_id": image_id,
                "image_src_path": src_img_path,
                "objects": objects
            }

    def map_category(self, original_category):
        return self.CATEGORY_MAP.get(original_category, None)

class CocoRoadWorkConverter(BaseConverter):
    """
    Converter for COCO-format ROADWork datasets (e.g., Pittsburgh subset).
    Expects a COCO JSON file with 'images', 'annotations', 'categories' fields.
    Uses the same category mapping as RoadWorkConverter.
    """

    def __init__(self, json_path, src_root, dst_root, split, **kwargs):
        """
        Args:
            json_path (Path): Path to the COCO JSON file (e.g., instances_train_pittsburgh_only.json).
            src_root (Path): Root directory containing the images (the 'file_name' in JSON is relative to this).
            dst_root (Path): Destination root for YOLO dataset.
            split (str): 'train' or 'val'.
        """
        super().__init__(src_root, dst_root, split, **kwargs)
        self.json_path = Path(json_path)
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)

        # Build image_id -> (width, height, file_name) mapping
        self.image_info = {}
        for img in self.data.get('images', []):
            img_id = img['id']
            self.image_info[img_id] = {
                'width': img['width'],
                'height': img['height'],
                'file_name': img['file_name'],
            }

        # Build category_id -> name mapping
        self.cat_id_to_name = {}
        for cat in self.data.get('categories', []):
            self.cat_id_to_name[cat['id']] = cat['name']

        # Build image_id -> list of (category_name, bbox) annotations
        self.image_annotations = {}
        for ann in self.data.get('annotations', []):
            img_id = ann['image_id']
            cat_name = self.cat_id_to_name.get(ann['category_id'], None)
            if cat_name is None:
                continue
            bbox = ann.get('bbox', None)
            if bbox and len(bbox) == 4:
                self.image_annotations.setdefault(img_id, []).append((cat_name, bbox))

    def parse_annotations(self):
        for img_id, info in self.image_info.items():
            image_id = Path(info['file_name']).stem
            src_img_path = self.src_root / info['file_name']
            objects = self.image_annotations.get(img_id, [])
            yield {
                'image_id': image_id,
                'image_src_path': src_img_path,
                'objects': objects,   # list of (category_name, bbox)
            }

    def map_category(self, original_category):
        # Use the same mapping as RoadWorkConverter (which is already defined in this file)
        # This map includes "Work Vehicle" -> 6 (construction_vehicle)
        return RoadWorkConverter.CATEGORY_MAP.get(original_category, None)