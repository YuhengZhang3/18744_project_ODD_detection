import os
import json
import glob
import random
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
    

def generate_clip_sensors(input_dir="../../source_images", json_dir="output_json"):
    os.makedirs(json_dir, exist_ok=True)
    
    # 1. Collect all images
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
 
    if not image_paths:
        print(f"No images found in '{input_dir}'.")
        return
 
    # 2. Setup Device and CLIP Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CLIP model on {device.type.upper()}...")
    
    # Use the official Hub ID instead of a local path
    model_id = "openai/clip-vit-large-patch14" 
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
 
    # 3. Define the visual proxies for our sensors (Removed Traffic and Location)
    categories = {
        "time": [
            "A dashcam photo taken at night in the pitch dark",
            "A dashcam photo taken in the early morning at sunrise",
            "A dashcam photo taken at midday with bright daylight",
            "A dashcam photo taken in the evening at sunset"
        ],
        "temperature": [
            "A dashcam photo of a freezing cold winter road with snow or ice",
            "A dashcam photo of a cool, crisp autumn road",
            "A dashcam photo of a warm, mild spring road",
            "A dashcam photo of a hot, blistering summer road"
        ],
        "humidity": [
            "A dashcam photo of a completely dry, clear atmosphere",
            "A dashcam photo of a hazy, humid atmosphere",
            "A dashcam photo of a wet, rainy atmosphere",
            "A dashcam photo of a very dense, foggy atmosphere"
        ]
    }
 
    print(f"Running CLIP inference to generate synthetic sensors for {len(image_paths)} images...")
 
    # 4. Flatten all prompts for single-pass inference
    category_names = list(categories.keys())
    all_prompts = []
    category_slices = {}  # maps category -> (start_idx, end_idx) in all_prompts
    for cat in category_names:
        start = len(all_prompts)
        all_prompts.extend(categories[cat])
        category_slices[cat] = (start, len(all_prompts))
    
    # Pre-tokenize text once (shared across all images)
    text_inputs = processor(text=all_prompts, return_tensors="pt", padding=True)
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)
    
    # Encode text features once (reused for every image)
    with torch.no_grad():
        text_outputs = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = model.text_projection(text_outputs.pooler_output)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            basename = os.path.splitext(filename)[0]
            
            try:
                img_pil = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
            
            # Encode image once
            img_inputs = processor(images=img_pil, return_tensors="pt")
            pixel_values = img_inputs["pixel_values"].to(device)
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            img_features = model.visual_projection(vision_outputs.pooler_output)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity with all 12 prompts at once
            similarity = (img_features @ text_features.T).squeeze(0)
            
            sensor_data = {}
            
            for category in category_names:
                start, end = category_slices[category]
                cat_logits = similarity[start:end]
                probs = cat_logits.softmax(dim=0).tolist()
                best_idx = probs.index(max(probs))
                
                # --- Map CLIP's visual guess to bounded numerical data ---
                
                if category == "time":
                    if best_idx == 0:   # Night
                        h = random.choice(list(range(20, 24)) + list(range(0, 5)))
                        m = random.randint(0, 59)
                    elif best_idx == 1: # Morning
                        h, m = random.randint(5, 9), random.randint(0, 59)
                    elif best_idx == 2: # Midday
                        h, m = random.randint(10, 15), random.randint(0, 59)
                    else:               # Evening
                        h, m = random.randint(16, 19), random.randint(0, 59)
                    sensor_data["clock_time"] = f"{h:02d}:{m:02d}"
                
                elif category == "temperature":
                    if best_idx == 0:   # Freezing
                        sensor_data["temperature_c"] = round(random.uniform(-15.0, 2.0), 1)
                    elif best_idx == 1: # Cool
                        sensor_data["temperature_c"] = round(random.uniform(3.0, 12.0), 1)
                    elif best_idx == 2: # Warm
                        sensor_data["temperature_c"] = round(random.uniform(13.0, 24.0), 1)
                    else:               # Hot
                        sensor_data["temperature_c"] = round(random.uniform(25.0, 40.0), 1)
                
                elif category == "humidity":
                    if best_idx == 0:   # Dry
                        sensor_data["humidity_pct"] = round(random.uniform(20.0, 45.0), 1)
                    elif best_idx == 1: # Hazy
                        sensor_data["humidity_pct"] = round(random.uniform(46.0, 70.0), 1)
                    elif best_idx == 2: # Wet
                        sensor_data["humidity_pct"] = round(random.uniform(71.0, 90.0), 1)
                    else:               # Foggy
                        sensor_data["humidity_pct"] = round(random.uniform(91.0, 100.0), 1)
 
            # 5. Save the inferred sensor data directly to JSON
            json_filename = f"{basename}.json"
            json_path = os.path.join(json_dir, json_filename)
            
            output_data = {
                "sensors": sensor_data
            }
            
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=4)
                
            print(f"Processed: {json_filename} | Time: {sensor_data['clock_time']} | Temp: {sensor_data['temperature_c']}°C | Humidity: {sensor_data['humidity_pct']}%")
 
if __name__ == "__main__":
    generate_clip_sensors()
