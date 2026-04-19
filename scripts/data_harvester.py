"""
Data Harvester for Stage-2 Tiebreaker MLP Training.

Reads stage-1 outputs (merged JSONs) + ground truth labels from three datasets
(BDD100K val, ACDC val, Roadwork) and assembles a clean [X, Y] tensor file.

Usage:
    python scripts/data_harvester.py \
        --bdd_outputs   stage1_outputs_BDD/merged_json \
        --bdd_labels    data/bdd100k_val/labels \
        --acdc_outputs  stage1_outputs_ACDC/merged_json \
        --acdc_root     data/ACDC_val \
        --rw_outputs    stage1_outputs_roadwork/merged_json \
        --rw_labels     data/roadwork_main/annotations/instances_train_gps_split.json \
                        data/roadwork_main/annotations/instances_val_gps_split.json \
        --output        data/tiebreaker_train.pt
"""

import os
import sys
import json
import glob
import argparse
import time as time_module
from pathlib import Path

import torch
import requests

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# ============================================================
# Constants
# ============================================================

X_DIM = 60
Y_DIM = 6  # fog, rain, snow, time, scene, anomalies

# Index mappings (must match training class order)
TIME_MAP = {"dawn/dusk": 0, "daytime": 1, "night": 2, "undefined": 3}
SCENE_MAP = {
    "city street": 0, "gas stations": 1, "highway": 2,
    "parking lot": 3, "residential": 4, "tunnel": 5, "undefined": 6,
}

# OSM one-hot: [City, Highway, Residential, Tunnel, GasStation, Parking]
OSM_LABELS = ["city", "highway", "residential", "tunnel", "gas_station", "parking"]

# ============================================================
# OSM Query (Overpass API) with caching
# ============================================================

_osm_cache = {}

def query_osm_context(lat, lon, radius=200):
    """
    Query Overpass API for road/POI types near (lat, lon).
    Returns 6-dim list: [City, Highway, Residential, Tunnel, GasStation, Parking]
    Each is 0 or 1. Returns all zeros on failure or unknown.
    """
    """This OSM is incredibly hard and unstable for accessing! Hardcoded to zeros."""
    return [0.0] * 6
    # cache_key = (round(lat, 4), round(lon, 4))
    # if cache_key in _osm_cache:
    #     return _osm_cache[cache_key]

    # one_hot = [0.0] * 6

    # query = f"""
    # [out:json][timeout:10];
    # (
    #   way["highway"](around:{radius},{lat},{lon});
    #   node["amenity"="fuel"](around:{radius},{lat},{lon});
    #   node["amenity"="parking"](around:{radius},{lat},{lon});
    # );
    # out tags;
    # """

    # try:
    #     resp = requests.post(
    #         "https://overpass-api.de/api/interpreter",
    #         data={"data": query},
    #         timeout=15,
    #     )
    #     resp.raise_for_status()
    #     elements = resp.json().get("elements", [])
    # except Exception:
    #     _osm_cache[cache_key] = one_hot
    #     return one_hot

    # for el in elements:
    #     tags = el.get("tags", {})
    #     hw = tags.get("highway", "")
    #     amenity = tags.get("amenity", "")
    #     tunnel = tags.get("tunnel", "")

    #     # Highway classification
    #     if hw in ("motorway", "motorway_link", "trunk", "trunk_link"):
    #         one_hot[1] = 1.0  # highway
    #     elif hw in ("primary", "secondary", "tertiary", "unclassified", "service"):
    #         one_hot[0] = 1.0  # city
    #     elif hw in ("residential", "living_street"):
    #         one_hot[2] = 1.0  # residential

    #     # Tunnel
    #     if tunnel == "yes":
    #         one_hot[3] = 1.0

    #     # Gas station
    #     if amenity == "fuel":
    #         one_hot[4] = 1.0

    #     # Parking
    #     if amenity == "parking":
    #         one_hot[5] = 1.0

    # _osm_cache[cache_key] = one_hot

    # # Rate limit: be polite to Overpass
    # time_module.sleep(0.1)
    # return one_hot


def save_osm_cache(path):
    """Save OSM cache to disk so we don't re-query on reruns."""
    serializable = {f"{k[0]},{k[1]}": v for k, v in _osm_cache.items()}
    with open(path, 'w') as f:
        json.dump(serializable, f)
    print(f"  OSM cache saved: {len(_osm_cache)} entries -> {path}")


def load_osm_cache(path):
    """Load OSM cache from disk."""
    global _osm_cache
    if os.path.exists(path):
        with open(path, 'r') as f:
            raw = json.load(f)
        _osm_cache = {tuple(map(float, k.split(","))): v for k, v in raw.items()}
        print(f"  OSM cache loaded: {len(_osm_cache)} entries from {path}")


# ============================================================
# GT Label Loaders
# ============================================================

def load_bdd_labels(label_dir):
    """
    Load BDD100K val labels.
    Returns: {image_basename_no_ext: {weather, scene, timeofday}}
    """
    labels = {}
    for jf in glob.glob(os.path.join(label_dir, "*.json")):
        with open(jf, 'r') as f:
            data = json.load(f)
        basename = os.path.splitext(data.get("name", os.path.basename(jf)))[0]
        attrs = data.get("attributes", {})
        labels[basename] = {
            "weather": attrs.get("weather", "undefined"),
            "scene": attrs.get("scene", "undefined"),
            "timeofday": attrs.get("timeofday", "undefined"),
        }
    print(f"  BDD labels loaded: {len(labels)} images")
    return labels


def load_acdc_labels(acdc_root):
    """
    Build ACDC label lookup from directory structure.
    Scans {acdc_root}/{condition}/{sequence}/*.png to map image basenames to conditions.
    Returns: {image_basename_no_ext: {condition: "fog"|"rain"|"snow"|"night"}}
    """
    labels = {}
    for condition in ["fog", "rain", "snow", "night"]:
        cond_dir = os.path.join(acdc_root, condition)
        if not os.path.isdir(cond_dir):
            continue
        for img_path in glob.glob(os.path.join(cond_dir, "**", "*.png"), recursive=True):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            labels[basename] = {"condition": condition}
    print(f"  ACDC labels loaded: {len(labels)} images")
    return labels


def load_roadwork_labels(json_paths):
    """
    Load Roadwork labels from one or more COCO-style annotation JSONs.
    Returns: {image_basename_no_ext: {weather, scene, daytime, travel_alteration}}
    """
    labels = {}
    for jp in json_paths:
        with open(jp, 'r') as f:
            data = json.load(f)
        for img_entry in data.get("images", []):
            basename = os.path.splitext(img_entry["file_name"])[0]
            tags = img_entry.get("scene_level_tags", {})
            labels[basename] = {
                "weather": tags.get("weather", []),       # list of strings
                "scene": tags.get("scene_environment", "Unknown"),
                "daytime": tags.get("daytime", "Unknown"),
                "travel_alteration": tags.get("travel_alteration", []),
            }
    print(f"  Roadwork labels loaded: {len(labels)} images")
    return labels


# ============================================================
# X Builder (60 dims)
# ============================================================

def _parse_clock_time(clock_str):
    """Convert "HH:MM" to normalized 0~1 float."""
    try:
        parts = clock_str.split(":")
        h, m = int(parts[0]), int(parts[1])
        return (h + m / 60.0) / 24.0
    except Exception:
        return 0.5  # default to noon


def _normalize_temperature(temp_c):
    """Normalize temperature from [-15, 40] to [0, 1]."""
    return max(0.0, min(1.0, (temp_c + 15.0) / 55.0))


def _normalize_humidity(hum_pct):
    """Normalize humidity from [0, 100] to [0, 1]."""
    return max(0.0, min(1.0, hum_pct / 100.0))


def _extract_yuheng_probabilities(yuheng_data, head_name):
    """
    Extract softmax probabilities from yuheng output.
    Handles both old format (yuheng.predictions.X.probabilities)
    and new format (yuheng.prediction.X.probabilities).
    For road_condition, prefers road_condition_infer in new format.
    """
    # Try new format first: yuheng.prediction
    pred = yuheng_data.get("prediction", None)
    if pred is not None:
        if head_name == "road_condition":
            # Prefer infer version in new format
            entry = pred.get("road_condition_infer", pred.get("road_condition_direct", None))
        else:
            entry = pred.get(head_name, None)
        if entry and "probabilities" in entry:
            return entry["probabilities"]

    # Fall back to old format: yuheng.predictions
    preds = yuheng_data.get("predictions", None)
    if preds is not None:
        entry = preds.get(head_name, None)
        if entry and "probabilities" in entry:
            return entry["probabilities"]

    return None


def build_x(merged, osm_cache_path=None):
    """
    Build 60-dim feature vector from a single merged JSON dict.
    Returns: list of 60 floats, or None if critical data is missing.
    """
    x = []

    # --- Environment (5 dims) ---
    # cloud_fraction
    cloud = merged.get("cloud_detection", {})
    x.append(cloud.get("cloud_fraction", 0.0))

    # fog, rain, snow severity
    weather = merged.get("weather", {})
    w = weather.get("weather", weather)  # handle nested {"weather": {...}}
    x.append(w.get("fog_severity", 0.0))
    x.append(w.get("rain_severity", 0.0))
    x.append(w.get("snow_severity", 0.0))

    # glare_ratio
    glare = merged.get("glare", {})
    x.append(glare.get("glare_ratio", 0.0))

    # --- Metadata softmax (45 dims) ---
    yuheng = merged.get("yuheng", {})

    for head, expected_dim in [
        ("time", 4), ("scene", 7), ("visibility", 3),
        ("anomalies", 4), ("road_condition", 27),
    ]:
        probs = _extract_yuheng_probabilities(yuheng, head)
        if probs and len(probs) == expected_dim:
            x.extend(probs)
        else:
            x.extend([0.0] * expected_dim)

    # --- Virtual sensors (10 dims) ---
    synth = merged.get("synth_outputs", {})
    sensors = synth.get("sensors", {})

    # temperature (1 dim, normalized)
    temp = sensors.get("temperature_c", 15.0)
    x.append(_normalize_temperature(temp))

    # humidity (1 dim, normalized)
    hum = sensors.get("humidity_pct", 50.0)
    x.append(_normalize_humidity(hum))

    # clock_time (1 dim, normalized)
    clock = sensors.get("clock_time", "12:00")
    x.append(_parse_clock_time(clock))

    # OSM map_context (6 dims)
    loc = sensors.get("location", {})
    lat = loc.get("lat", None)
    lon = loc.get("lon", None)
    if lat is not None and lon is not None:
        # print(f"querying {lat} and {lon}")
        osm = query_osm_context(lat, lon)
    else:
        osm = [0.0] * 6
    x.extend(osm)


    # GeoCLIP confidence (1 dim) — DEAD FEATURE
    # x.append(loc.get("geoclip_confidence", 0.0))
    # Originally intended to gate OSM 6-dim one-hot, but OSM queries fail from
    # lightning.ai. With OSM stuck at zeros, this confidence carries no useful
    # signal on its own. Kept as 0 to preserve X_DIM=60 ABI with existing ckpt.
    x.append(0.0)

    assert len(x) == X_DIM, f"Expected {X_DIM} dims, got {len(x)}"
    return x


# ============================================================
# Y Builders (6 dims each, with -1 masking)
# ============================================================

def build_y_bdd(gt):
    """
    Build Y vector from BDD100K GT.
    Y = [fog, rain, snow, time, scene, anomalies]
    """
    y = [0.0] * Y_DIM

    # Weather -> fog/rain/snow binary
    weather = gt["weather"].lower()
    y[0] = 1.0 if weather == "foggy" else 0.0
    y[1] = 1.0 if weather == "rainy" else 0.0
    y[2] = 1.0 if weather == "snowy" else 0.0

    # Time of day
    tod = gt["timeofday"].lower()
    if tod in TIME_MAP:
        y[3] = float(TIME_MAP[tod])
    else:
        y[3] = -1.0

    # Scene
    scene = gt["scene"].lower()
    if scene in SCENE_MAP:
        y[4] = float(SCENE_MAP[scene])
    else:
        y[4] = -1.0

    # Anomalies: BDD has no GT
    y[5] = -1.0

    return y


def build_y_acdc(gt):
    """
    Build Y vector from ACDC GT.
    Y = [fog, rain, snow, time, scene, anomalies]
    """
    y = [-1.0] * Y_DIM
    cond = gt["condition"]

    # Weather
    if cond == "fog":
        y[0], y[1], y[2] = 1.0, 0.0, 0.0
    elif cond == "rain":
        y[0], y[1], y[2] = 0.0, 1.0, 0.0
    elif cond == "snow":
        y[0], y[1], y[2] = 0.0, 0.0, 1.0
    elif cond == "night":
        # Clear night: no fog/rain/snow
        y[0], y[1], y[2] = 0.0, 0.0, 0.0
        y[3] = float(TIME_MAP["night"])  # time = night

    # Scene: ACDC has no scene GT
    y[4] = -1.0

    # Anomalies: ACDC has no anomaly GT
    y[5] = -1.0

    return y


# Roadwork mapping tables
_RW_DAYTIME_MAP = {
    "light": TIME_MAP["daytime"],
    "dark": TIME_MAP["night"],
    "twilight": TIME_MAP["dawn/dusk"],
}

_RW_SCENE_MAP = {
    "urban": SCENE_MAP["city street"],
    "suburban": SCENE_MAP["undefined"], #NOTE: BDD does not have a built-in suburban scene label
    "highway": SCENE_MAP["highway"],
    "rural": SCENE_MAP["undefined"],
}


def build_y_roadwork(gt):
    """
    Build Y vector from Roadwork GT.
    Y = [fog, rain, snow, time, scene, anomalies]
    """
    y = [-1.0] * Y_DIM

    # Weather: list of strings, e.g. ["Rain"], ["Partly Cloudy"]
    weather_list = [w.lower() for w in gt.get("weather", [])]

    if "unknown" in weather_list or len(weather_list) == 0:
        # Can't determine weather
        y[0], y[1], y[2] = -1.0, -1.0, -1.0
    else:
        has_rain = "rain" in weather_list
        has_snow = "snow" in weather_list
        # No fog category in Roadwork
        y[0] = -1.0  # fog: unknown
        y[1] = 1.0 if has_rain else 0.0
        y[2] = 1.0 if has_snow else 0.0

    # Time of day
    daytime = gt.get("daytime", "Unknown").lower()
    if daytime in _RW_DAYTIME_MAP:
        y[3] = float(_RW_DAYTIME_MAP[daytime])
    else:
        y[3] = -1.0

    # Scene
    scene = gt.get("scene", "Unknown").lower()
    if scene in _RW_SCENE_MAP:
        y[4] = float(_RW_SCENE_MAP[scene])
    elif scene in ("other", "unknown"):
        y[4] = -1.0
    else:
        y[4] = -1.0

    # Anomalies: TODO - map travel_alteration when anomaly GT is available
    y[5] = -1.0

    return y


# ============================================================
# Main Harvester
# ============================================================

def harvest_dataset(merged_dir, gt_labels, dataset_name, build_y_fn):
    """
    Process one dataset: read merged JSONs, build X and Y.
    Returns: (X_list, Y_list, skipped_count)
    """
    json_files = sorted(glob.glob(os.path.join(merged_dir, "*.json")))
    X_list, Y_list = [], []
    skipped = 0

    for jf in json_files:
        basename = os.path.splitext(os.path.basename(jf))[0]

        # Check if we have GT for this image
        if basename not in gt_labels:
            skipped += 1
            continue

        # Load merged JSON
        with open(jf, 'r') as f:
            merged = json.load(f)

        # Build X
        x = build_x(merged)
        if x is None:
            skipped += 1
            continue

        # Build Y
        y = build_y_fn(gt_labels[basename])

        X_list.append(x)
        Y_list.append(y)

    print(f"  {dataset_name}: {len(X_list)} samples harvested, {skipped} skipped")
    return X_list, Y_list


def main():
    parser = argparse.ArgumentParser(description="Data Harvester for Stage-2 Tiebreaker MLP")
    
    # BDD
    parser.add_argument("--bdd_outputs", type=str, default="stage1_outputs_BDD/merged_json",
                        help="Path to BDD merged JSON directory")
    parser.add_argument("--bdd_labels", type=str, default="data/bdd100k_val/labels",
                        help="Path to BDD label JSON directory")
    
    # ACDC
    parser.add_argument("--acdc_outputs", type=str, default="stage1_outputs_ACDC/merged_json",
                        help="Path to ACDC merged JSON directory")
    parser.add_argument("--acdc_root", type=str, default="data/ACDC_val",
                        help="Path to ACDC root with fog/rain/snow/night subdirs")
    
    # Roadwork
    parser.add_argument("--rw_outputs", type=str, default="stage1_outputs_roadwork/merged_json",
                        help="Path to Roadwork merged JSON directory")
    parser.add_argument("--rw_labels", type=str, nargs="+",
                        default=[
                            "data/roadwork_main/annotations/instances_train_gps_split.json",
                            "data/roadwork_main/annotations/instances_val_gps_split.json",
                        ],
                        help="Path(s) to Roadwork annotation JSON(s)")
    
    # Output
    parser.add_argument("--output", type=str, default="data/tiebreaker_train.pt",
                        help="Output .pt file path")
    
    # OSM
    parser.add_argument("--osm_cache", type=str, default="data/osm_cache.json",
                        help="Path to OSM query cache file")
    parser.add_argument("--skip_osm", action="store_true",
                        help="Skip OSM queries (fill with zeros)")
    
    args = parser.parse_args()

    # Resolve paths relative to project root
    def resolve(p):
        if os.path.isabs(p):
            return p
        return os.path.join(project_root, p)

    print("=" * 50)
    print("  DATA HARVESTER - Stage 2 Tiebreaker")
    print("=" * 50)

    # Load OSM cache
    if not args.skip_osm:
        load_osm_cache(resolve(args.osm_cache))
    else:
        print("  OSM queries SKIPPED (--skip_osm)")
        # Monkey-patch to return zeros
        global query_osm_context
        _original_osm = query_osm_context
        query_osm_context = lambda lat, lon, radius=200: [0.0] * 6

    all_X, all_Y = [], []

    # --- BDD ---
    print("\n[1/3] Processing BDD100K val...")
    bdd_labels = load_bdd_labels(resolve(args.bdd_labels))
    bdd_x, bdd_y = harvest_dataset(
        resolve(args.bdd_outputs), bdd_labels, "BDD", build_y_bdd
    )
    all_X.extend(bdd_x)
    all_Y.extend(bdd_y)

    # --- ACDC ---
    print("\n[2/3] Processing ACDC val...")
    acdc_labels = load_acdc_labels(resolve(args.acdc_root))
    acdc_x, acdc_y = harvest_dataset(
        resolve(args.acdc_outputs), acdc_labels, "ACDC", build_y_acdc
    )
    all_X.extend(acdc_x)
    all_Y.extend(acdc_y)

    # --- Roadwork ---
    print("\n[3/3] Processing Roadwork...")
    rw_label_paths = [resolve(p) for p in args.rw_labels]
    rw_labels = load_roadwork_labels(rw_label_paths)
    rw_x, rw_y = harvest_dataset(
        resolve(args.rw_outputs), rw_labels, "Roadwork", build_y_roadwork
    )
    all_X.extend(rw_x)
    all_Y.extend(rw_y)

    # --- Save ---
    if not all_X:
        print("\nERROR: No samples harvested! Check paths and data.")
        return

    X_tensor = torch.tensor(all_X, dtype=torch.float32)
    Y_tensor = torch.tensor(all_Y, dtype=torch.float32)

    # Save OSM cache
    if not args.skip_osm:
        save_osm_cache(resolve(args.osm_cache))

    output_path = resolve(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({"X": X_tensor, "Y": Y_tensor}, output_path)

    print("\n" + "=" * 50)
    print(f"  HARVEST COMPLETE")
    print(f"  Total samples: {len(all_X)}")
    print(f"  X shape: {X_tensor.shape}")
    print(f"  Y shape: {Y_tensor.shape}")
    print(f"  Saved to: {output_path}")
    print("=" * 50)

    # --- Stats ---
    print("\nLabel coverage stats:")
    y_names = ["fog", "rain", "snow", "time", "scene", "anomalies"]
    for i, name in enumerate(y_names):
        col = Y_tensor[:, i]
        valid = (col != -1).sum().item()
        masked = (col == -1).sum().item()
        print(f"  {name:12s}: {valid:6d} valid, {masked:6d} masked")


if __name__ == "__main__":
    main()