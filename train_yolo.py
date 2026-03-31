#!/usr/bin/env python3
"""
Training script for YOLO model on mixed ROADWork and BDD100K dataset.
Performs two-stage fine-tuning: first freeze backbone, then full network.
"""

import argparse
import sys
from pathlib import Path
import re
from models.yolo_model import YOLOModel
from configs.yolo_config import yolo_config
from typing import Optional


project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model for ODD perception.')
    parser.add_argument('--stage2only', action='store_true',
                        help='If set, run only stage2 using stage1/best.pt weights (must exist).')
    return parser.parse_args()

def find_latest_run_dir(base_path: Path, experiment_name: str) -> Optional[Path]:
    """
    Finds the latest run directory for a given experiment name.
    Handles cases like 'exp_name', 'exp_name2', 'exp_name3'.
    """
    parent_dir = base_path.parent
    if not parent_dir.is_dir():
        return None

    # Regex to match 'experiment_name' or 'experiment_name#'
    pattern = re.compile(rf"^{re.escape(experiment_name)}(\d*)$")
    
    matching_dirs = []
    for d in parent_dir.iterdir():
        if d.is_dir():
            match = pattern.match(d.name)
            if match:
                suffix = match.group(1)
                num = int(suffix) if suffix else 0
                matching_dirs.append((num, d))
    
    if not matching_dirs:
        return None
    
    # Sort by number and return the path of the highest one
    latest_dir = sorted(matching_dirs, key=lambda x: x[0], reverse=True)[0][1]
    return latest_dir


def main():
    args = parse_args()

    print("Starting YOLO training with configuration:")
    for key, value in yolo_config.items():
        print(f"  {key}: {value}")

    # custom kw arguments that YOLO dont recognize
    custom_keys = [
        'model_name', 'device', 
        'stage1_epochs', 'freeze', 'lr0_stage1', 'stage1_name',
        'stage2_epochs', 'lr0_stage2', 'stage2_name',
        'project'
    ]
    
    # YOLO supported hyperparams
    yolo_kwargs = {k: v for k, v in yolo_config.items() if k not in custom_keys}

    base_project_dir = Path(yolo_config.get('project') or 'runs/detect') # Default to runs/detect if project is None
    stage1_base_name = yolo_config.get('stage1_name', 'yolo11/stage1')
    stage1_full_base_path = base_project_dir / stage1_base_name # This is the "project/name" part passed to YOLO

    if args.stage2only:
        # ---------- Stage 2 only: load weights directly ----------
        # Find the actual latest output directory for stage 1
        latest_stage1_run_dir = find_latest_run_dir(stage1_full_base_path, stage1_full_base_path.name)
        
        if not latest_stage1_run_dir:
            print(f"Error: No completed Stage 1 run found at {stage1_full_base_path.parent / stage1_full_base_path.name}* to resume from.")
            sys.exit(1)
            
        weights_path = latest_stage1_run_dir / 'weights' / 'best.pt'
        if not weights_path.exists():
            print(f"Error: Weights file not found in the latest stage 1 run: {weights_path}")
            sys.exit(1)
            
        print(f"\n=== Stage 2 only: loading weights from {weights_path} ===")
        model = YOLOModel(model_name=str(weights_path), device=yolo_config.get('device', 'cuda'))
    else:
        # ---------- Full two-stage training: run stage1 first ----------
        print("\n=== Stage 1: Training with frozen backbone ===")
        model = YOLOModel(
            model_name=yolo_config.get('model_name', 'yolo11m.pt'),
            device=yolo_config.get('device', 'cuda')
        )
        stage1_save_dir = model.train( # Capture the actual save directory
            epochs=yolo_config.get('stage1_epochs', 12),
            freeze=yolo_config.get('freeze', 10),
            lr0=yolo_config.get('lr0_stage1', 0.00038),
            project=yolo_config.get('project'),
            name=stage1_full_base_path.name, # Pass only the name part
            **yolo_kwargs # inject remaining YOLO parameters
        )

        # Use the captured save_dir to construct the path for stage 2
        weights_path = stage1_save_dir / 'weights' / 'best.pt'
        if not weights_path.exists():
            raise RuntimeError(f"Error: Stage1 best weights not found at {weights_path}. Cannot proceed to stage 2.")
        print(f"\n=== Stage 2: Full fine-tuning, loading weights from {weights_path} ===")
        model = YOLOModel(model_name=str(weights_path), device=yolo_config.get('device', 'cuda'))

    # ---------- Stage 2 (common for both paths) ----------
    model.train(
        epochs=yolo_config.get('stage2_epochs', 15),
        freeze=0,  # unfreeze all layers
        lr0=yolo_config.get('lr0_stage2', 0.00038),
        project=yolo_config.get('project'),
        name=yolo_config.get('stage2_name', 'yolo11/stage2'), # YOLO will handle its numbering if already exists
        resume=False,
        **yolo_kwargs
    )

    print("\nTraining completed successfully.")

if __name__ == '__main__':
    main()