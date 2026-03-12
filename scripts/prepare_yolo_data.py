#!/usr/bin/env python3
"""
Prepare YOLO training data from ROADWork and BDD100K.
Usage: python scripts/prepare_yolo_data.py
"""

import random
from pathlib import Path


from yolo_style_label_converter import RoadWorkConverter, BDDConverter


random.seed(42)


project_root = Path(__file__).parent.parent
data_root = project_root / "data"

# ---------- ROADWork ----------
roadwork_src_root = data_root / "roadwork"          # images/  traj_annotations/
roadwork_train_json = roadwork_src_root / "traj_annotations" / "trajectories_train_equidistant.json"
roadwork_val_json = roadwork_src_root / "traj_annotations" / "trajectories_val_equidistant.json"


# bdd100k val set, I split this further into training set and validating set for yolo
bdd_src_root = data_root / "bdd100k_val"            # images/  labels/
bdd_json_dir = bdd_src_root / "labels"              # JSON 


yolo_root = data_root / "yolo"

# 1. ROADWork train
if roadwork_train_json.exists():
    print("Converting ROADWork train...")
    converter = RoadWorkConverter(
        json_path=roadwork_train_json,
        src_root=roadwork_src_root,
        dst_root=yolo_root,
        split="train",
        use_symlink=True
    )
    converter.convert()
    print("ROADWork train done.")

# 2. ROADWork val
if roadwork_val_json.exists():
    print("Converting ROADWork val...")
    converter = RoadWorkConverter(
        json_path=roadwork_val_json,
        src_root=roadwork_src_root,
        dst_root=yolo_root,
        split="val",
        use_symlink=True
    )
    converter.convert()
    print("ROADWork val done.")

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

# 4. data.yaml
yaml_path = yolo_root / "data.yaml"
with open(yaml_path, 'w') as f:
    f.write(f"""# YOLO dataset configuration
path: {yolo_root.resolve()}  # dataset root dir
train: images/train
val: images/val
nc: 6
names: ['pedestrian', 'bicycle', 'vehicle', 'construction_channelizer', 'construction_barrier', 'construction_sign']
""")
print(f"data.yaml created at {yaml_path}")