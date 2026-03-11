import os
import cv2
import json
import torch
import numpy as np
import glob
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

def process_clouds(input_dir="../../source_images", output_dir="output_boxes", json_dir="output_json"):
    # 1. Setup directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_dir, "*.[jp][pn]*[g]")) 

    if not image_paths:
        print(f"No images found in {input_dir}.")
        return

    # 2. Load Model & Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SegFormer on {device.upper()}...")
    
    model_id = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
    model.eval()

    SKY_CLASS_ID = 10 
    print(f"Processing {len(image_paths)} images for cloud coverage...")

    # 3. Process each image
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = img_rgb.shape[:2]

        # --- RUN SEGFORMER INFERENCE ---
        inputs = processor(images=img_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=(original_h, original_w), 
            mode="bilinear", 
            align_corners=False
        )
        
        predictions = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        sky_mask = (predictions == SKY_CLASS_ID).astype(np.uint8) * 255

        # --- CLOUD DETECTION LOGIC ---
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1] # Saturation
        V = hsv[:, :, 2] # Brightness
        
        sky_pixels_bool = (sky_mask == 255)
        total_sky_pixels = np.sum(sky_pixels_bool)
        
        cloud_fraction = 0.0
        cloud_mask = np.zeros_like(S)
        
        if total_sky_pixels > 0:
            # Clouds are areas in the sky with low saturation (white/gray) and high brightness
            cloud_pixels_bool = sky_pixels_bool & (S < 80) & (V > 100)
            cloud_mask[cloud_pixels_bool] = 255
            cloud_fraction = float(np.sum(cloud_pixels_bool) / total_sky_pixels)

        # --- SAVE METRICS TO JSON ---
        json_filename = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(json_dir, json_filename)
        
        results_dict = {
            "cloud_fraction": round(cloud_fraction, 4)
        }
        
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)

        # --- VISUALIZATION AUGMENTATIONS ---
        overlay = img_bgr.copy()
        
        # 1. Add a black mask to the sky mask to dim non-sky pixels
        overlay[sky_mask != 255] = [0, 0, 0]
        
        # Apply the combined overlay
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)

        # 3. Find contours of the sky mask to draw bounding boxes
        contours, _ = cv2.findContours(sky_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw the red box around the sky area
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # --- SAVE AUGMENTED IMAGE ---
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, img_bgr)
        print(f"Saved: {out_path} | Cloud Cover: {cloud_fraction * 100:.1f}%")

if __name__ == "__main__":
    process_clouds(input_dir="../../source_images", output_dir="output_boxes", json_dir="output_json")