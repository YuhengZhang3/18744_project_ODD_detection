"""
Configuration for YOLO training on mixed ROADWork + BDD dataset.
This dictionary follows the parameters expected by ultralytics.YOLO.train().
Based on YOLO26 training recipe for medium model.
"""

yolo_config = {
    # -------------------- Model and data --------------------
    'model_name': 'yolo11m.pt',          # Pre-trained weights: medium version
    'data': 'data/yolo_augmented/data.yaml',   # Dataset config file (now using augmented data)
    'device': 'cuda',                     # Use GPU
    'cache': True,
    'patience': 10,

    # -------------------- Training hyperparameters --------------------
    'batch': 64,                         # Batch size per GPU (YOLO26 default)
    'imgsz': 640,                         # Input image size
    'workers': 8,                        # Number of dataloader workers

    # Stage 1: Freeze backbone
    'stage1_epochs': 20,                  # Number of epochs for frozen stage (can be adjusted)
    'freeze': 10,                         # Freeze first 10 layers (backbone)
    'lr0_stage1': 0.00038,                # YOLO26 M learning rate
    'stage1_name': 'yolo26/stage1',       # Experiment subfolder name

    # Stage 2: Full fine-tuning
    'stage2_epochs': 20,                  # Additional epochs for full fine-tuning (YOLO26 M 80 total? but we do 20+60=80)
    'lr0_stage2': 0.00038,                # Keep same LR (since we fine-tune full model)
    'stage2_name': 'yolo26/stage2',       # Experiment subfolder name

    # Common optimizer settings (YOLO26 M uses MuSGD, which is SGD with momentum)
    'optimizer': 'SGD',                   # MuSGD = SGD with momentum
    'momentum': 0.948,                    # YOLO26 M momentum
    'weight_decay': 0.00027,              # YOLO26 M weight decay

    # Learning rate schedule
    'lrf': 0.882,                         # Final LR factor (LR = lr0 * lrf at end)
    'warmup_epochs': 0.99,                # YOLO26 M warmup epochs
    'warmup_momentum': 0.8,               # Warmup initial momentum (keep default)
    'warmup_bias_lr': 0.1,                # Warmup initial bias LR (default)

    # Data augmentation (YOLO26 M values)
    'hsv_h': 0.013,                       # HSV-Hue augmentation
    'hsv_s': 0.353,                       # HSV-Saturation augmentation
    'hsv_v': 0.194,                       # HSV-Value augmentation
    'degrees': 0.0,                       # Rotation (YOLO26 M uses ~0)
    'translate': 0.275,                   # Translation
    'scale': 0.95,                        # Scaling
    'shear': 0.0,                         # Shear (YOLO26 M uses ~0)
    'perspective': 0.0,                   # Perspective (not used)
    'flipud': 0.0,                        # Vertical flip (not common)
    'fliplr': 0.304,                      # Horizontal flip probability
    'mosaic': 0.992,                      # Mosaic augmentation probability
    'mixup': 0.427,                       # Mixup probability
    'copy_paste': 0.304,                  # Copy-paste probability (for instance segmentation? but keep)
    'bgr': 0.0,                           # BGR augmentation probability (YOLO26 M uses 0)

    # Loss weights (YOLO26 M)
    'box': 9.83,                          # Box loss gain
    'cls': 0.65,                          # Class loss gain
    'dfl': 0.96,                          # DFL loss gain

    # -------------------- Experiment tracking --------------------
    'project': None,                      # Root directory for logs and checkpoints
    'seed': 42,                           # Random seed for reproducibility
}