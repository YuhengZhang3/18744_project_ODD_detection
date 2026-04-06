import os
import json
import cv2
import glob
import numpy as np
from tqdm import tqdm

def get_near_far_contrast(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    x0 = int(w * 0.25)
    x1 = int(w * 0.75)

    y0_near = int(h * 0.5)
    y1_near = int(h * 0.9)
    near = img_gray[y0_near:y1_near, x0:x1]

    y0_far = int(h * 0.2)
    y1_far = int(h * 0.5)
    far = img_gray[y0_far:y1_far, x0:x1]

    near_c = float(near.std()) / 255.0
    far_c = float(far.std()) / 255.0

    return near_c, far_c


def assign_visibility(weather, tod, near_c, far_c):
    # 2 = good, 1 = medium, 0 = poor
    fog_near_thr = 0.135
    fog_far_thr = 0.17
    night_far_thr = 0.19

    if weather == "foggy":
        return 0

    if near_c < fog_near_thr and far_c < fog_far_thr:
        return 0

    if tod == "night" and far_c < night_far_thr and near_c >= fog_near_thr:
        return 1

    return 2


def predict_visibility(input_dir="../../source_images", json_dir="output_json"):
    # Create the output directory if it doesn't exist
    os.makedirs(json_dir, exist_ok=True)
    
    # Grab all images safely
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        print(f"No images found in '{input_dir}'.")
        return

    print(f"Processing {len(image_paths)} images for visibility contrast...")

    for img_path in tqdm(image_paths):
        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]
        
        img = cv2.imread(img_path)
        if img is None:
            continue

        json_filename = f"{basename}.json"
        json_path = os.path.join(json_dir, json_filename)
        
        # Load existing JSON data if it exists (to pull 'weather' or 'timeofday' if already predicted)
        existing_data = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                pass
                
        # Try to extract weather and time of day from previous pipeline data; fallback to undefined
        weather = existing_data.get("weather", "undefined")
        tod = existing_data.get("timeofday", existing_data.get("lighting_condition", "undefined"))

        # Calculate contrast metrics
        near_c, far_c = get_near_far_contrast(img)
        vis = assign_visibility(weather, tod, near_c, far_c)

        # Merge new metrics into the dictionary
        existing_data["visibility_score"] = int(vis)
        existing_data["near_contrast"] = round(near_c, 4)
        existing_data["far_contrast"] = round(far_c, 4)

        # Save back to JSON
        with open(json_path, "w") as f:
            json.dump(existing_data, f, indent=4)


if __name__ == "__main__":
    predict_visibility()