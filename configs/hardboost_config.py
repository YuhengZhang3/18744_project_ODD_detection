# hard class boost settings used by hardboost / alternating training

# mode names
NORMAL_MODE = "normal"
HARD_MODE = "hard"

# epoch-level alternating pattern
ALTERNATING_PATTERN = [NORMAL_MODE, HARD_MODE]

# normal mode: no extra boost
TIME_NORMAL_BOOST = {}
SCENE_NORMAL_BOOST = {}
VIS_NORMAL_BOOST = {}
ROAD_NORMAL_BOOST = {}

# hard mode: targeted boost for weak classes
TIME_HARD_BOOST = {
    0: 2.0,   # dawn/dusk
}

SCENE_HARD_BOOST = {
    0: 1.6,   # city street
}

VIS_HARD_BOOST = {
    1: 1.5,   # medium
}

ROAD_HARD_BOOST = {
    7: 2.0,   # dry_mud
    10: 2.5,  # water_asphalt_smooth
    13: 2.5,  # water_concrete_smooth
    16: 2.2,  # water_gravel
    17: 2.2,  # water_mud
    18: 2.8,  # wet_asphalt_smooth
    19: 2.8,  # wet_asphalt_slight
    21: 2.8,  # wet_concrete_smooth
    22: 2.8,  # wet_concrete_slight
    24: 2.2,  # wet_gravel
    25: 2.2,  # wet_mud
}

BOOST_MAP = {
    NORMAL_MODE: {
        "time": TIME_NORMAL_BOOST,
        "scene": SCENE_NORMAL_BOOST,
        "visibility": VIS_NORMAL_BOOST,
        "road": ROAD_NORMAL_BOOST,
    },
    HARD_MODE: {
        "time": TIME_HARD_BOOST,
        "scene": SCENE_HARD_BOOST,
        "visibility": VIS_HARD_BOOST,
        "road": ROAD_HARD_BOOST,
    },
}
