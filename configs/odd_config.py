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
    "drivable": {
        "type": "segmentation",
        # keep 3 classes for now:
        # 0 = background, 1 = alternative_drivable, 2 = direct_drivable
        "num_classes": 3,
        "loss_weight": 1.0,
    },
}