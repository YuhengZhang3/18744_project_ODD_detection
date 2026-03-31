#!/usr/bin/env python3
"""
Prepare YOLO training data from ROADWork and BDD100K.
Usage: python scripts/prepare_yolo_data.py
"""

import random
from pathlib import Path
import json

from yolo_style_label_converter import RoadWorkConverter, BDDConverter, CocoRoadWorkConverter


random.seed(42)


project_root = Path(__file__).parent.parent
data_root = project_root / "data"

# ---------- ROADWork ----------
roadwork_traj_src_root = data_root / "roadwork_traj"          # images/  traj_annotations/
roadwork_traj_train_json = roadwork_traj_src_root / "traj_annotations" / "trajectories_train_equidistant.json"
roadwork_traj_val_json = roadwork_traj_src_root / "traj_annotations" / "trajectories_val_equidistant.json"


# bdd100k val set, I split this further into training set and validating set for yolo
bdd_src_root = data_root / "bdd100k_val"            # images/  labels/
bdd_json_dir = bdd_src_root / "labels"              # JSON 


yolo_root = data_root / "yolo"

# 1. ROADWork train
if roadwork_traj_train_json.exists():
    print("Converting ROADWork train...")
    converter = RoadWorkConverter(
        json_path=roadwork_traj_train_json,
        src_root=roadwork_traj_src_root,
        dst_root=yolo_root,
        split="train",
        use_symlink=True
    )
    converter.convert()
    print("ROADWork traj train done.")

# 2. ROADWork val
if roadwork_traj_val_json.exists():
    print("Converting ROADWork val...")
    converter = RoadWorkConverter(
        json_path=roadwork_traj_val_json,
        src_root=roadwork_traj_src_root,
        dst_root=yolo_root,
        split="val",
        use_symlink=True
    )
    converter.convert()
    print("ROADWork traj val done.")

# 3. BDD: 8000 train + 2000 val
if bdd_json_dir.exists():
    # get all JSON
    all_json_files = list(bdd_json_dir.glob("*.json"))
    print(f"Found {len(all_json_files)} BDD JSON files.")
    if len(all_json_files) == 0:
        print("No JSON files found, skip BDD.")
    else:
        random.shuffle(all_json_files)
        val_count = 2000
        train_files = all_json_files[val_count:]
        val_files = all_json_files[:val_count]
        print(f"BDD train: {len(train_files)}, val: {len(val_files)}")

        # BDD training
        if train_files:
            print("Converting BDD train...")
            converter = BDDConverter(
                src_root=bdd_src_root,
                dst_root=yolo_root,
                split="train",
                json_files=train_files,
                use_symlink=True
            )
            converter.convert()
            print("BDD train done.")

        # BDD validating
        if val_files:
            print("Converting BDD val...")
            converter = BDDConverter(
                src_root=bdd_src_root,
                dst_root=yolo_root,
                split="val",
                json_files=val_files,
                use_symlink=True
            )
            converter.convert()
            print("BDD val done.")
else:
    print(f"BDD directory not found: {bdd_json_dir}")

# 4. BDD bikes, for simplicity, add all to training set
# 4. BDD 自行车图片（划分 train/val）
bdd_bikes_root = data_root / "bdd100k_bikes"
bdd_bikes_json_dir = bdd_bikes_root / "labels"
if bdd_bikes_json_dir.exists():
    bike_json_files = list(bdd_bikes_json_dir.glob("*.json"))
    if bike_json_files:
        print(f"Found {len(bike_json_files)} BDD bike images.")
        random.shuffle(bike_json_files)
        val_count = min(1000, len(bike_json_files))
        val_files = bike_json_files[:val_count]
        train_files = bike_json_files[val_count:]
        print(f"Assigning {len(train_files)} to train, {len(val_files)} to val.")

        if train_files:
            converter = BDDConverter(
                src_root=bdd_bikes_root,
                dst_root=yolo_root,
                split="train",
                json_files=train_files,
                use_symlink=True
            )
            converter.convert()
        if val_files:
            converter = BDDConverter(
                src_root=bdd_bikes_root,
                dst_root=yolo_root,
                split="val",
                json_files=val_files,
                use_symlink=True
            )
            converter.convert()
        print("BDD bikes added.")

# 5. Pittsburgh ROADWork COCO dataset (train/val)
pittsburgh_root = data_root / "roadwork_main"
pittsburgh_train_json = pittsburgh_root / "annotations" / "instances_train_pittsburgh_only.json"
pittsburgh_val_json = pittsburgh_root / "annotations" / "instances_val_pittsburgh_only.json"
pittsburgh_img_root = pittsburgh_root / "images"

if pittsburgh_train_json.exists():
    print("Converting Pittsburgh train...")
    converter = CocoRoadWorkConverter(
        json_path=pittsburgh_train_json,
        src_root=pittsburgh_img_root,
        dst_root=yolo_root,
        split="train",
        use_symlink=True
    )
    converter.convert()
    print("Pittsburgh train done.")

if pittsburgh_val_json.exists():
    print("Converting Pittsburgh val...")
    converter = CocoRoadWorkConverter(
        json_path=pittsburgh_val_json,
        src_root=pittsburgh_img_root,
        dst_root=yolo_root,
        split="val",
        use_symlink=True
    )
    converter.convert()
    print("Pittsburgh val done.")


# 6. data.yaml
yaml_path = yolo_root / "data.yaml"
with open(yaml_path, 'w') as f:
    f.write(f"""# YOLO dataset configuration
path: {yolo_root.resolve()}  # dataset root dir
train: images/train
val: images/val
nc: 7
names: ['pedestrian', 'bicycle', 'vehicle', 'construction_channelizer', 'construction_barrier', 'construction_sign', 'construction_vehicle']
""")

# 7. After all conversions, collect all ROADWork stems from original JSON files
def collect_roadwork_stems_from_json(json_paths, src_root):
    stems = set()
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
        # ROADWork traj format (list of entries with 'image' field)
        if isinstance(data, list):
            for entry in data:
                image_rel = entry.get('image')
                if image_rel:
                    stem = Path(image_rel).stem
                    stems.add(stem)
        # COCO format (with 'images' list)
        elif 'images' in data:
            for img in data['images']:
                file_name = img.get('file_name')
                if file_name:
                    stem = Path(file_name).stem
                    stems.add(stem)
    return stems

roadwork_json_paths = [
    roadwork_traj_train_json,
    roadwork_traj_val_json,
    pittsburgh_train_json,
    pittsburgh_val_json
]
# Only include paths that exist
existing_json_paths = [p for p in roadwork_json_paths if p.exists()]
if existing_json_paths:
    stems = collect_roadwork_stems_from_json(existing_json_paths, roadwork_traj_src_root)
    # Write to file
    roadwork_stems_file = yolo_root / 'roadwork_stems.txt'
    with open(roadwork_stems_file, 'w') as f:
        for stem in sorted(stems):
            f.write(stem + '\n')
    print(f"ROADWork stems saved to {roadwork_stems_file} ({len(stems)} entries)")