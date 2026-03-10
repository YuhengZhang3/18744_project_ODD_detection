#!/usr/bin/env python3
"""
Training script for YOLO model on mixed ROADWork and BDD100K dataset.
Performs two-stage fine-tuning: first freeze backbone, then full network.
"""

import argparse
import sys
from pathlib import Path
from models.yolo_model import YOLOModel
from configs.yolo_config import yolo_config


project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model for ODD perception.')
    parser.add_argument('--stage2only', action='store_true',
                        help='If set, run only stage2 using stage1/best.pt weights (must exist).')
    return parser.parse_args()


def main():
    args = parse_args()

    print("Starting YOLO training with configuration:")
    for key, value in yolo_config.items():
        print(f"  {key}: {value}")

    if args.stage2only:
        # ---------- Stage 2 only: load weights directly ----------
        # When project=None, default base is 'runs/detect'
        weights_path = Path('runs/detect') / yolo_config.get('stage1_name', 'stage1') / 'weights' / 'best.pt'
        if not weights_path.exists():
            print(f"Error: Weights file not found: {weights_path}")
            sys.exit(1)
        print(f"\n=== Stage 2 only: loading weights from {weights_path} ===")
        model = YOLOModel(model_name=str(weights_path), device=yolo_config.get('device', 'cuda'))
    else:
        # ---------- Full two-stage training: run stage1 first ----------
        print("\n=== Stage 1: Training with frozen backbone ===")
        model = YOLOModel(
            model_name=yolo_config.get('model_name', 'yolov8m.pt'),
            device=yolo_config.get('device', 'cuda')
        )
        model.train(
            data=yolo_config['data'],
            epochs=yolo_config.get('stage1_epochs', 30),
            batch=yolo_config['batch'],
            imgsz=yolo_config['imgsz'],
            freeze=yolo_config.get('freeze', 10),
            lr0=yolo_config.get('lr0_stage1', 0.001),
            project=yolo_config['project'],  # None
            name=yolo_config.get('stage1_name', 'stage1'),
            workers=yolo_config.get('workers', 8),
            optimizer=yolo_config.get('optimizer', 'SGD'),
            seed=yolo_config.get('seed', 42)
        )

        # Prepare stage2 weights path from stage1 result (actual location)
        weights_path = Path('runs/detect') / yolo_config.get('stage1_name', 'stage1') / 'weights' / 'best.pt'
        if not weights_path.exists():
            raise RuntimeError(f"Error: Stage1 best weights not found at {weights_path}. Cannot proceed to stage 2.")
        print(f"\n=== Stage 2: Full fine-tuning, loading weights from {weights_path} ===")
        model = YOLOModel(model_name=str(weights_path), device=yolo_config.get('device', 'cuda'))

    # ---------- Stage 2 (common for both paths) ----------
    model.train(
        data=yolo_config['data'],
        epochs=yolo_config.get('stage2_epochs', 70),
        batch=yolo_config['batch'],
        imgsz=yolo_config['imgsz'],
        freeze=0,  # unfreeze all layers
        lr0=yolo_config.get('lr0_stage2', 0.0001),
        project=yolo_config['project'],  # None
        name=yolo_config.get('stage2_name', 'stage2'),
        workers=yolo_config.get('workers', 8),
        optimizer=yolo_config.get('optimizer', 'SGD'),
        seed=yolo_config.get('seed', 42),
        resume=False
    )

    print("\nTraining completed successfully.")


if __name__ == '__main__':
    main()