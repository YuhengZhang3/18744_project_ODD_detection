# scripts/infer_single.py
"""
Run full stage-1 + stage-2 pipeline on a single image.

Usage:
    python scripts/infer_single.py path/to/image.jpg
    python scripts/infer_single.py path/to/image.jpg --skip_stage1  # if merged JSON already exists
"""

import os
import sys
import json
import shutil
import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run_stage1(image_path, work_dir):
    """Run all 6 stage-1 models on a single image via temp directory."""
    from models.cloud_detection.clouds import process_clouds
    from models.glare.test_glare import evaluate_test_set
    from models.weather.infer import predict_weather
    from utils.infer_api import load_pipeline, infer_path
    from models.yolo.traffic_workzone import process_traffic_workzone
    from models.synth.location import append_geoclip_location
    from models.synth.synth_data import generate_clip_sensors

    # Temp input dir with just this one image
    img_dir = os.path.join(work_dir, "input")
    os.makedirs(img_dir, exist_ok=True)
    dst = os.path.join(img_dir, os.path.basename(image_path))
    shutil.copy2(image_path, dst)

    out_dir = os.path.join(work_dir, "outputs")

    # [0] Synth sensors
    print("[0/5] Synth sensors...")
    synth_dir = os.path.join(out_dir, "synth_outputs")
    generate_clip_sensors(input_dir=img_dir, json_dir=synth_dir)
    append_geoclip_location(input_dir=img_dir, json_dir=synth_dir)

    # [1] Cloud detection
    print("[1/5] Cloud detection...")
    cloud_dir = os.path.join(out_dir, "cloud_detection")
    boxes_dir = os.path.join(work_dir, "output_boxes")
    process_clouds(input_dir=img_dir, output_dir=boxes_dir, json_dir=cloud_dir)

    # [2] Glare
    print("[2/5] Glare...")
    glare_dir = os.path.join(out_dir, "glare")
    glare_model = str(PROJECT_ROOT / "models" / "glare" / "custom_glare_model")
    evaluate_test_set(model_dir=glare_model, test_dir=img_dir, output_dir=glare_dir)

    # [3] Weather
    print("[3/5] Weather...")
    weather_dir = os.path.join(out_dir, "weather")
    weather_weights = str(PROJECT_ROOT / "models" / "weather" / "weather_resnet18_best.pth")
    predict_weather(input_dir=img_dir, json_dir=weather_dir, model_path=weather_weights)

    # [4] ODD model
    print("[4/5] ODD model...")
    yuheng_dir = os.path.join(out_dir, "yuheng")
    os.makedirs(yuheng_dir, exist_ok=True)
    odd_ckpt = str(PROJECT_ROOT / "models" / "yuheng" / "odd_full_infer_best.pt")
    pipeline = load_pipeline(ckpt_path=odd_ckpt)
    results = infer_path(input_path=img_dir, pipeline=pipeline)
    for item in results:
        stem = os.path.splitext(item["image_name"])[0]
        with open(os.path.join(yuheng_dir, f"{stem}.json"), "w") as f:
            json.dump(item, f, indent=2)

    # [5] YOLO
    print("[5/5] YOLO...")
    yolo_dir = os.path.join(out_dir, "yolo")
    process_traffic_workzone(
        input_dir=img_dir,
        json_dir=yolo_dir,
        model_path="models/yolo/18744_project_ODD_detection_runs_detect_yolo_stage2_weights_best.pt",
        thresholds_path="models/yolo/density_thresholds.json",
    )

    return out_dir


def merge_jsons(out_dir, basename):
    """Merge all sub-model JSONs for one image."""
    merged = {}
    subdirs = ["synth_outputs", "cloud_detection", "glare", "weather", "yuheng", "yolo"]
    for sub in subdirs:
        jf = os.path.join(out_dir, sub, f"{basename}.json")
        if os.path.exists(jf):
            with open(jf, "r") as f:
                merged[sub] = json.load(f)
    return merged


def run_stage2(merged):
    """Run MLP + Guardrail on a merged JSON dict."""
    from scripts.data_harvester import build_x
    from models.tiebreaker.tiebreaker_mlp import TiebreakerMLP
    from models.tiebreaker.tiebreakers_guard import run_guardrail, ROAD_COND_NAMES

    # Build X
    x = build_x(merged)
    x_tensor = torch.tensor([x], dtype=torch.float32)

    # Load MLP
    model = TiebreakerMLP()
    ckpt_path = str(PROJECT_ROOT / "checkpoints_tiebreaker" / "tiebreaker_best.pt")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        out = model(x_tensor)

    s2_fog = float(torch.sigmoid(out["fog"]))
    s2_rain = float(torch.sigmoid(out["rain"]))
    s2_snow = float(torch.sigmoid(out["snow"]))
    s2_time = int(out["time"].argmax(dim=1))
    s2_scene = int(out["scene"].argmax(dim=1))

    TIME_NAMES = ["dawn/dusk", "daytime", "night", "undefined"]
    SCENE_NAMES = ["city street", "gas stations", "highway",
                   "parking lot", "residential", "tunnel", "undefined"]

    # Guardrail
    pred = merged.get("yuheng", {}).get("prediction", {})
    rc = pred.get("road_condition_infer") or pred.get("road_condition_direct")
    vis = pred.get("visibility")

    guardrail_result = None
    if rc and vis:
        s1_road27 = np.array(rc["probabilities"], dtype=np.float32)
        s1_vis3 = np.array(vis["probabilities"], dtype=np.float32)
        guardrail_result = run_guardrail(s1_road27, s1_vis3, s2_fog, s2_rain, s2_snow)

    return {
        "weather": {"fog": s2_fog, "rain": s2_rain, "snow": s2_snow},
        "time": TIME_NAMES[s2_time],
        "scene": SCENE_NAMES[s2_scene],
        "guardrail": guardrail_result,
    }


def print_comparison(merged, s2_result):
    """Print stage-1 vs stage-2 side by side."""
    from models.tiebreaker.tiebreakers_guard import ROAD_COND_NAMES, ROAD_COND_STATES

    print("\n" + "=" * 70)
    print(f"  {'':30s} {'STAGE-1':>15s}   {'STAGE-2':>15s}")
    print("=" * 70)

    # Weather
    w = merged.get("weather", {})
    w = w.get("weather", w)
    s2w = s2_result["weather"]
    for key in ["fog", "rain", "snow"]:
        s1v = w.get(f"{key}_severity", 0.0)
        s2v = s2w[key]
        flag = " <<<" if abs(s2v - s1v) > 0.3 else ""
        print(f"  {key:30s} {s1v:15.3f}   {s2v:15.3f}{flag}")

    # Time
    s1_time = merged.get("yuheng", {}).get("prediction", {}).get("time", {}).get("label", "?")
    print(f"  {'time':30s} {s1_time:>15s}   {s2_result['time']:>15s}")

    # Scene
    s1_scene = merged.get("yuheng", {}).get("prediction", {}).get("scene", {}).get("label", "?")
    print(f"  {'scene':30s} {s1_scene:>15s}   {s2_result['scene']:>15s}")

    # Road condition
    g = s2_result.get("guardrail")
    if g is not None:
        pred = merged["yuheng"]["prediction"]
        rc = pred.get("road_condition_infer") or pred.get("road_condition_direct")
        s1_top = ROAD_COND_NAMES[int(np.argmax(rc["probabilities"]))]
        s2_top = ROAD_COND_NAMES[int(np.argmax(g["road_condition_corrected"]))]
        print(f"  {'road_condition top-1':30s} {s1_top:>15s}   {s2_top:>15s}")

        print(f"\n  {'5-state aggregate':30s} {'STAGE-1':>15s}   {'STAGE-2':>15s}")
        s1_agg = {}
        s1_road = np.array(rc["probabilities"], dtype=np.float32)
        for state, idxs in ROAD_COND_STATES.items():
            s1_agg[state] = float(s1_road[idxs].sum())
        s2_agg = g["road_condition_aggregated"]
        for state in ["dry", "wet", "water", "snow", "ice"]:
            print(f"  {state:30s} {s1_agg[state]:15.3f}   {s2_agg[state]:15.3f}")

        # Visibility
        vis = pred.get("visibility", {})
        s1_vis = vis.get("probabilities", [0, 0, 0])
        s2_vis = g["visibility_corrected"]
        for i, name in enumerate(["poor", "medium", "good"]):
            print(f"  {'visibility.' + name:30s} {s1_vis[i]:15.3f}   {float(s2_vis[i]):15.3f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Single-image inference pipeline")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--skip_stage1", action="store_true",
                        help="Skip stage-1, use existing merged JSON from --merged_json")
    parser.add_argument("--merged_json", default=None,
                        help="Path to existing merged JSON (use with --skip_stage1)")
    parser.add_argument("--keep_workdir", action="store_true",
                        help="Don't delete temp work directory")
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    if args.skip_stage1:
        # Load existing merged JSON
        if args.merged_json:
            mj = args.merged_json
        else:
            # Try common locations
            for candidate in [
                f"stage1_outputs_BDD/merged_json/{basename}.json",
                f"stage1_outputs_ACDC/merged_json/{basename}.json",
                f"stage1_outputs_roadwork/merged_json/{basename}.json",
                f"stage1_outputs/merged_json/{basename}.json",
            ]:
                full = str(PROJECT_ROOT / candidate)
                if os.path.exists(full):
                    mj = full
                    break
            else:
                print(f"ERROR: No merged JSON found for {basename}")
                return

        print(f"Loading existing merged JSON: {mj}")
        with open(mj, "r") as f:
            merged = json.load(f)
    else:
        print(f"Running full stage-1 pipeline on: {image_path}")
        work_dir = tempfile.mkdtemp(prefix="odd_single_")
        print(f"Work dir: {work_dir}")
        try:
            out_dir = run_stage1(image_path, work_dir)
            merged = merge_jsons(out_dir, basename)
            # Save merged for reuse
            merged_out = str(PROJECT_ROOT / f"single_inference_{basename}.json")
            with open(merged_out, "w") as f:
                json.dump(merged, f, indent=2)
            print(f"Merged JSON saved: {merged_out}")
        finally:
            if not args.keep_workdir:
                shutil.rmtree(work_dir, ignore_errors=True)

    # Stage 2
    print("\nRunning stage-2 (MLP + Guardrail)...")
    s2 = run_stage2(merged)
    print_comparison(merged, s2)


if __name__ == "__main__":
    main()