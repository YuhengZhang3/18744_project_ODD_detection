# config for odd model

BACKBONE_NAME = "dinov2_vitl14"
ADAPTER_DIM = 1024

# change num_classes or add new heads here.
# for new global heads using cls feature, also add the name to _GLOBAL_HEADS in models/odd_model.py.

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
}