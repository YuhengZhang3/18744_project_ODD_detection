import os
import sys
import json
 
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
 
def run_pipeline():
    # Define your shared paths here so they are easy to update
    INPUT_IMAGES = os.path.join(project_root, "source_images_ACDC")
    OUTPUT_JSON = os.path.join(project_root, "stage1_outputs_")
    OUTPUT_BOXES = os.path.join(project_root, "models", "cloud_detection", "output_boxes")
    
    print("========================================")
    print("   STARTING VISION ANALYSIS PIPELINE    ")
    print("========================================")
    
    # ---------------------------------------------------------
    # 0. SYNTH SENSOR DATA
    # ---------------------------------------------------------
    # print("\n[0/5] Synthesizing sensor data...")
    # synth_output_dir = os.path.join(OUTPUT_JSON, "synth_outputs")
 
    # generate_clip_sensors(
    #     input_dir=INPUT_IMAGES, 
    #     json_dir=synth_output_dir
    # )
    # append_geoclip_location(
    #     input_dir=INPUT_IMAGES,
    #     json_dir=synth_output_dir
    # )
 
    # ---------------------------------------------------------
    # 1. CLOUD DETECTION
    # ---------------------------------------------------------
    # print("\n[1/5] Running Cloud Detection...")
    # cloud_output_dir = os.path.join(OUTPUT_JSON, "cloud_detection")
 
    # process_clouds(
    #     input_dir=INPUT_IMAGES,
    #     output_dir=OUTPUT_BOXES,
    #     json_dir=cloud_output_dir
    # )
 
    # ---------------------------------------------------------
    # 2. GLARE EVALUATION
    # ---------------------------------------------------------
    # print("\n[2/5] Running Glare Evaluation...")
    # glare_model = os.path.join(project_root, "models", "glare", "custom_glare_model")
    # glare_output_dir = os.path.join(OUTPUT_JSON, "glare")
    
    # evaluate_test_set(
    #     model_dir=glare_model,
    #     test_dir=INPUT_IMAGES,
    #     output_dir=glare_output_dir
    # )
 
    # ---------------------------------------------------------
    # 3. WEATHER PREDICTION
    # ---------------------------------------------------------
    # print("\n[3/5] Running Weather Prediction...")
    # weather_weights = os.path.join(project_root, "models", "weather", "weather_resnet18_best.pth")
    # weather_output_dir = os.path.join(OUTPUT_JSON, "weather")
    
    # predict_weather(
    #     input_dir=INPUT_IMAGES,
    #     json_dir=weather_output_dir,
    #     model_path=weather_weights
    # )
    
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
    # print("\n[5/5] Running YOLO Evaluation...")
    # yolo_output_dir = os.path.join(OUTPUT_JSON, "yolo")
    
    # process_traffic_workzone(
    #     input_dir=INPUT_IMAGES,
    #     json_dir=yolo_output_dir,
    #     model_path="models/yolo/18744_project_ODD_detection_runs_detect_yolo_stage2_weights_best.pt",
    #     thresholds_path="models/yolo/density_thresholds.json"
    # )
 
    print("\n========================================")
    print("         PIPELINE COMPLETE!             ")
    print("========================================")
 
if __name__ == "__main__":
    run_pipeline()