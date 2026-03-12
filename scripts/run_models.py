import os
import sys

# Add the project root (one directory up from /scripts) to the Python path
# so we can import from the 'models' directory cleanly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import your model functions
from models.cloud_detection.clouds import process_clouds
from models.glare.test_glare import evaluate_test_set
from models.precip_model.infer import predict_weather
from models.yuheng.run_predictions import run_predictions
from models.yolo.traffic_workzone import process_traffic_workzone

def run_pipeline():
    # Define your shared paths here so they are easy to update
    INPUT_IMAGES = os.path.join(project_root, "source_images")
    OUTPUT_JSON = os.path.join(project_root, "outputs")
    OUTPUT_BOXES = os.path.join(project_root, "models", "cloud_detection", "output_boxes")
    
    print("========================================")
    print("   STARTING VISION ANALYSIS PIPELINE    ")
    print("========================================")

    # ---------------------------------------------------------
    # 1. CLOUD DETECTION
    # ---------------------------------------------------------
    print("\n[1/5] Running Cloud Detection...")
    cloud_output_dir = os.path.join(OUTPUT_JSON, "cloud_detection")

    process_clouds(
        input_dir=INPUT_IMAGES,
        output_dir=OUTPUT_BOXES,
        json_dir=cloud_output_dir
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
        output_dir=glare_output_dir
    )

    # ---------------------------------------------------------
    # 3. WEATHER PREDICTION
    # ---------------------------------------------------------
    print("\n[3/5] Running Weather Prediction...")
    weather_weights = os.path.join(project_root, "models", "precip_model", "weather_resnet18_best.pth")
    weather_output_dir = os.path.join(OUTPUT_JSON, "weather")
    
    predict_weather(
        input_dir=INPUT_IMAGES,
        json_dir=weather_output_dir,
        model_path=weather_weights
    )
    
    # ---------------------------------------------------------
    # 4. BDD100k EVALUATION
    # ---------------------------------------------------------
    print("\n[4/5] Running BDD100k Evaluation...")
    yuheng_output_dir = os.path.join(OUTPUT_JSON, "yuheng")
    
    run_predictions(
        source_directory=INPUT_IMAGES,
        output_json_directory=yuheng_output_dir,
        checkpoint_path="models/yuheng/stage2_best.pt",
        drivable_label_directory=None,
        limit=None
    )
    
    # ---------------------------------------------------------
    # 5. YOLO
    # ---------------------------------------------------------
    print("\n[5/5] Running YOLO Evaluation...")
    yolo_output_dir = os.path.join(OUTPUT_JSON, "yolo")
    
    process_traffic_workzone(
        input_dir=INPUT_IMAGES,
        json_dir=yolo_output_dir,
        model_path="models/yolo/18744_project_ODD_detection_runs_detect_yolo_stage2_weights_best.pt",
        thresholds_path="models/yolo/density_thresholds.json"
    )

    print("\n========================================")
    print("         PIPELINE COMPLETE!             ")
    print("========================================")

if __name__ == "__main__":
    run_pipeline()