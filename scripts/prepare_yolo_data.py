#!/usr/bin/env python3
"""
Prepare YOLO training data from ROADWork and BDD100K.
Usage: python scripts/prepare_yolo_data.py
"""

import random
from pathlib import Path

# 导入我们之前定义的转换器
from yolo_style_label_converter import RoadWorkConverter, BDDConverter

# 设置随机种子保证可复现
random.seed(42)

# 项目根目录（脚本位于 scripts/ 下，所以上级是项目根）
project_root = Path(__file__).parent.parent
data_root = project_root / "data"

# ---------- ROADWork 转换 ----------
roadwork_src_root = data_root / "roadwork"          # 包含 images/ 和 traj_annotations/
roadwork_train_json = roadwork_src_root / "traj_annotations" / "trajectories_train_equidistant.json"
roadwork_val_json = roadwork_src_root / "traj_annotations" / "trajectories_val_equidistant.json"

# BDD 数据路径（val 全部图片，我们将划分一部分用于验证）
bdd_src_root = data_root / "bdd100k_val"            # 包含 images/ 和 labels/
bdd_json_dir = bdd_src_root / "labels"              # 所有 JSON 文件在此

# YOLO 格式输出根目录
yolo_root = data_root / "yolo"

# 1. 转换 ROADWork 训练集
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

# 2. 转换 ROADWork 验证集
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

# 3. 处理 BDD 数据：划分 8000 训练 + 2000 验证
if bdd_json_dir.exists():
    # 获取所有 JSON 文件
    all_json_files = list(bdd_json_dir.glob("*.json"))
    print(f"Found {len(all_json_files)} BDD JSON files.")
    if len(all_json_files) == 0:
        print("No JSON files found, skip BDD.")
    else:
        # 随机打乱
        random.shuffle(all_json_files)
        # 划分：前 2000 作为验证，其余作为训练
        val_count = 2000
        train_files = all_json_files[val_count:]
        val_files = all_json_files[:val_count]
        print(f"BDD train: {len(train_files)}, val: {len(val_files)}")

        # 转换训练部分（放入 YOLO train 目录）
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

        # 转换验证部分（放入 YOLO val 目录）
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

# 4. 生成 data.yaml
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