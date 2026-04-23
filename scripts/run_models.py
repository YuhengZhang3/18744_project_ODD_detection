import os
import sys
import json
import glob

# Add the project root (one directory up from /scripts) to the Python path
# so we can import from the 'models' directory cleanly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
 
# Import your model functions
from models.cloud_detection.clouds import process_clouds
from models.glare.test_glare import evaluate_test_set
from models.weather.infer import predict_weather
from utils.infer_api import load_pipeline, infer_path
from models.yolo.traffic_workzone import process_traffic_workzone
from models.synth.location import append_geoclip_location
from models.synth.synth_data import generate_clip_sensors
from scripts.data_harvester import build_x
from models.tiebreaker.tiebreaker_mlp import TiebreakerMLP
from models.tiebreaker.tiebreakers_guard import run_guardrail, ROAD_COND_NAMES, aggregate_road_states

 
def run_pipeline():
    # Define your shared paths here so they are easy to update
    INPUT_IMAGES = os.path.join(project_root, "source_images")
    OUTPUT_JSON = os.path.join(project_root, "outputs")
    OUTPUT_BOXES = os.path.join(project_root, "models", "cloud_detection", "output_boxes")
    
    print("========================================")
    print("   STARTING VISION ANALYSIS PIPELINE    ")
    print("========================================")
    
    # ---------------------------------------------------------
    # 0. SYNTH SENSOR DATA
    # ---------------------------------------------------------
    print("\n[0/5] Synthesizing sensor data...")
    synth_output_dir = os.path.join(OUTPUT_JSON, "synth_outputs")
 
    generate_clip_sensors(
        input_dir=INPUT_IMAGES, 
        json_dir=synth_output_dir
    )
    append_geoclip_location(
        input_dir=INPUT_IMAGES,
        json_dir=synth_output_dir
    )
 
    # ---------------------------------------------------------
    # 1. CLOUD DETECTION
    # ---------------------------------------------------------
    print("\n[1/5] Running Cloud Detection...")
    cloud_output_dir = os.path.join(OUTPUT_JSON, "cloud_detection")
 
    process_clouds(
        input_dir=INPUT_IMAGES,
        output_dir=OUTPUT_BOXES,
        json_dir=cloud_output_dir,
        save_vis=True
    )
 
    # ---------------------------------------------------------
    # 2. GLARE EVALUATION
    # ---------------------------------------------------------
    print("\n[2/5] Running Glare Evaluation...")
    glare_model = os.path.join(project_root, "models", "glare", "custom_glare_model")
    glare_output_dir = os.path.join(OUTPUT_JSON, "glare")
    
    evaluate_test_set(
        model_dir=glare_model,
        test_dir=INPUT_IMAGES,
        output_dir=glare_output_dir, 
        save_mask=True
    )
 
    # ---------------------------------------------------------
    # 3. WEATHER PREDICTION
    # ---------------------------------------------------------
    print("\n[3/5] Running Weather Prediction...")
    weather_weights = os.path.join(project_root, "models", "weather", "weather_resnet18_best.pth")
    weather_output_dir = os.path.join(OUTPUT_JSON, "weather")
    
    predict_weather(
        input_dir=INPUT_IMAGES,
        json_dir=weather_output_dir,
        model_path=weather_weights
    )
    
    # ---------------------------------------------------------
    # 4. ODD MODEL (DINOv2 + all heads)
    # ---------------------------------------------------------
    print("\n[4/5] Running ODD Model Evaluation...")
    yuheng_output_dir = os.path.join(OUTPUT_JSON, "yuheng")
    os.makedirs(yuheng_output_dir, exist_ok=True)
    
    odd_ckpt = os.path.join(project_root, "models", "yuheng", "odd_full_infer_best.pt")
    pipeline = load_pipeline(ckpt_path=odd_ckpt)
    results = infer_path(input_path=INPUT_IMAGES, pipeline=pipeline)
    
    # Save flat (one JSON per image) to stay compatible with join_jsons.py
    for item in results:
        stem = os.path.splitext(item["image_name"])[0]
        out_path = os.path.join(yuheng_output_dir, f"{stem}.json")
        with open(out_path, "w") as f:
            json.dump(item, f, indent=2)
    
    print(f"  ODD model: {len(results)} images processed")
    
    # ---------------------------------------------------------
    # 5. YOLO
    # ---------------------------------------------------------
    print("\n[5/5] Running YOLO Evaluation...")
    yolo_output_dir = os.path.join(OUTPUT_JSON, "yolo")
    
    process_traffic_workzone(
        input_dir=INPUT_IMAGES,
        json_dir=yolo_output_dir,
        model_path="models/yolo/yolo_traffic_workzone.pt",
        thresholds_path="models/yolo/density_thresholds.json"
    )

    # ---------------------------------------------------------
    # 6. JOIN JSONS
    # ---------------------------------------------------------
    print("\n[6/7] Merging stage-1 outputs...")
    merged_dir = os.path.join(OUTPUT_JSON, "merged_json")
    os.makedirs(merged_dir, exist_ok=True)

    # Collect all sub-model output dirs
    sub_dirs = [
        os.path.join(OUTPUT_JSON, sub) for sub in
        ["synth_outputs", "cloud_detection", "glare", "weather", "yuheng", "yolo"]
    ]

    # Merge by matching filenames across sub-dirs
    all_basenames = set()
    for sd in sub_dirs:
        if os.path.isdir(sd):
            for f in os.listdir(sd):
                if f.endswith(".json"):
                    all_basenames.add(os.path.splitext(f)[0])

    merge_count = 0
    for basename in sorted(all_basenames):
        merged = {}
        for sd in sub_dirs:
            jf = os.path.join(sd, f"{basename}.json")
            if os.path.exists(jf):
                sub_name = os.path.basename(sd)
                with open(jf, 'r') as f:
                    merged[sub_name] = json.load(f)
        out_path = os.path.join(merged_dir, f"{basename}.json")
        with open(out_path, 'w') as f:
            json.dump(merged, f, indent=2)
        merge_count += 1

    print(f"  Merged {merge_count} images -> {merged_dir}")

    # ---------------------------------------------------------
    # 7. STAGE-2: MLP Tiebreaker + Guardrail
    # ---------------------------------------------------------
    print("\n[7/7] Running Stage-2 (MLP Tiebreaker + Guardrail)...")

    import numpy as np
    import torch

    # Load MLP
    mlp_ckpt = os.path.join(project_root, "models", "tiebreaker", "tiebreaker_best.pt")
    if not os.path.exists(mlp_ckpt):
        print(f"  WARNING: MLP checkpoint not found at {mlp_ckpt}, skipping stage-2")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TiebreakerMLP()
        state = torch.load(mlp_ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        # Batch process all merged JSONs
        merged_files = sorted(glob.glob(os.path.join(merged_dir, "*.json")))
        all_merged = []
        valid_indices = []

        for i, mf in enumerate(merged_files):
            with open(mf, 'r') as f:
                merged = json.load(f)
            x = build_x(merged)
            if x is not None:
                all_merged.append((mf, merged))
                valid_indices.append(len(all_merged) - 1)

        if not all_merged:
            print("  WARNING: No valid samples for stage-2")
        else:
            # Batched MLP forward
            X = torch.tensor(
                [build_x(m) for _, m in all_merged],
                dtype=torch.float32
            ).to(device)

            with torch.no_grad():
                out = model(X)

            fog_probs  = torch.sigmoid(out["fog"]).cpu().numpy()
            rain_probs = torch.sigmoid(out["rain"]).cpu().numpy()
            snow_probs = torch.sigmoid(out["snow"]).cpu().numpy()
            time_preds = out["time"].cpu().argmax(dim=1).numpy()
            scene_preds = out["scene"].cpu().argmax(dim=1).numpy()
            anomaly_preds = out["anomalies"].cpu().argmax(dim=1).numpy()

            TIME_NAMES = ["dawn/dusk", "daytime", "night", "undefined"]
            SCENE_NAMES = ["city street", "gas stations", "highway",
                           "parking lot", "residential", "tunnel", "undefined"]
            ANOMALY_NAMES = ["none", "extreme_weather", "road_blockage_hazard", "other"]

            corrected_count = 0
            for i, (mf, merged) in enumerate(all_merged):
                p_fog = float(fog_probs[i])
                p_rain = float(rain_probs[i])
                p_snow = float(snow_probs[i])

                # Get stage-1 softmax for Guardrail
                pred = merged.get("yuheng", {}).get("prediction", {})
                rc = pred.get("road_condition_infer") or pred.get("road_condition_direct")
                vis = pred.get("visibility")

                guardrail_result = None
                if rc and vis:
                    s1_road27 = np.array(rc["probabilities"], dtype=np.float32)
                    s1_vis3 = np.array(vis["probabilities"], dtype=np.float32)
                    if s1_road27.shape == (27,) and s1_vis3.shape == (3,):
                        guardrail_result = run_guardrail(
                            s1_road27, s1_vis3, p_fog, p_rain, p_snow
                        )

                # Build stage-2 correction block
                stage2 = {
                    "weather_corrected": {
                        "fog": p_fog,
                        "rain": p_rain,
                        "snow": p_snow,
                    },
                    "time_corrected": {
                        "label": TIME_NAMES[int(time_preds[i])],
                        "class_id": int(time_preds[i]),
                    },
                    "scene_corrected": {
                        "label": SCENE_NAMES[int(scene_preds[i])],
                        "class_id": int(scene_preds[i]),
                    },
                    "anomalies_corrected": {
                        "label": ANOMALY_NAMES[int(anomaly_preds[i])],
                        "class_id": int(anomaly_preds[i]),
                    },
                }

                if guardrail_result is not None:
                    rc_corr = guardrail_result["road_condition_corrected"]
                    vis_corr = guardrail_result["visibility_corrected"]
                    rc_top = int(np.argmax(rc_corr))
                    vis_top = int(np.argmax(vis_corr))

                    stage2["road_condition_corrected"] = {
                        "label": ROAD_COND_NAMES[rc_top],
                        "class_id": rc_top,
                        "confidence": float(rc_corr[rc_top]),
                        "probabilities": rc_corr.tolist(),
                    }
                    stage2["road_condition_aggregated"] = guardrail_result["road_condition_aggregated"]
                    stage2["visibility_corrected"] = {
                        "label": ["poor", "medium", "good"][vis_top],
                        "class_id": vis_top,
                        "confidence": float(vis_corr[vis_top]),
                        "probabilities": vis_corr.tolist(),
                    }

                # Write back: add stage2 block, don't touch anything else
                merged["stage2"] = stage2
                with open(mf, 'w') as f:
                    json.dump(merged, f, indent=2)
                corrected_count += 1

            print(f"  Stage-2 corrections applied to {corrected_count} images")
 
    print("\n========================================")
    print("         PIPELINE COMPLETE!             ")
    print("========================================")
 
if __name__ == "__main__":
    run_pipeline()