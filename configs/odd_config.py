# config for odd model

BACKBONE_NAME = "dinov2_vitl14"
ADAPTER_DIM = 1024

# change num_classes or add new heads here.
# for new global heads using cls feature, also add the name to _GLOBAL_HEADS in models/odd_model.py.

ODD_HEAD_CONFIG = {
    "weather": {
        "type": "multiclass",
        # e.g. clear / cloudy / rain / snow / other
        "num_classes": 5,
        "loss_weight": 1.0,
    },
    "lighting": {
        "type": "multiclass",
        # e.g. dark / medium / bright (from light_level / visibility / glare)
        "num_classes": 3,
        "loss_weight": 1.0,
    },
    "time": {
        "type": "multiclass",
        # e.g. day / dawn_dusk / night
        "num_classes": 3,
        "loss_weight": 1.0,
    },
    "traffic": {
        "type": "multiclass",
        # e.g. none / low / medium / high from car/ped/bicycle
        "num_classes": 4,
        "loss_weight": 1.0,
    },
    "road_condition": {
        "type": "multiclass",
        # e.g. dry / wet / snow / ice / unpaved
        "num_classes": 5,
        "loss_weight": 1.0,
    },
    "scene": {
        "type": "multiclass",
        # e.g. city / suburban / residential / highway / parking / underpass
        "num_classes": 6,
        "loss_weight": 1.0,
    },
    "anomalies": {
        "type": "multiclass",
        # e.g. none / sandstorm / tornado / hurricane / collapsed_road
        "num_classes": 5,
        "loss_weight": 1.0,
    },
}