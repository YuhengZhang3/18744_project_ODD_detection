# config for odd model

BACKBONE_NAME = "dinov2_vitl14"
ADAPTER_DIM = 1024

ODD_HEAD_CONFIG = {
    "time": {
        "type": "multiclass",
        "num_classes": 4,
        "loss_weight": 1.0,
    },
    "scene": {
        "type": "multiclass",
        "num_classes": 7,
        "loss_weight": 1.0,
    },
    "visibility": {
        "type": "multiclass",
        # 0 = poor, 1 = medium, 2 = good
        "num_classes": 3,
        "loss_weight": 1.0,
    },
    "road_condition": {
        "type": "multiclass",
        # 27 classes in RSCD
        "num_classes": 27,
        "loss_weight": 1.0,
    },
    "road_state": {
        "type": "multiclass",
        # 0=dry, 1=wet, 2=water, 3=snow, 4=ice
        "num_classes": 5,
        "loss_weight": 1.0,
    },
    "road_severity": {
        "type": "multiclass",
        # 0=none, 1=smooth, 2=slight, 3=severe
        "num_classes": 4,
        "loss_weight": 1.0,
    },
    "drivable": {
        "type": "segmentation",
        # keep 3 classes for now:
        # 0 = background, 1 = alternative_drivable, 2 = direct_drivable
        "num_classes": 3,
        "loss_weight": 1.0,
    },
}