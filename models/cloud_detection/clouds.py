import os
import cv2
import json
import torch
import numpy as np
import glob
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

def process_images(input_dir="../../source_images", output_dir="output_boxes", json_dir="output_json"):
    # 1. Setup directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_dir, "*.[jp][pn]*[g]")) # matches jpg, jpeg, png

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

    # In Cityscapes, class 10 is 'sky'
    SKY_CLASS_ID = 10 

    print(f"Processing {len(image_paths)} images...")

    # 3. Process each image
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        
        # Read image using OpenCV (BGR format)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        # Convert to RGB for SegFormer
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = img_rgb.shape[:2]

        # 4. Run Inference
        inputs = processor(images=img_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            
        # 5. Upsample logits to original image size
        logits = outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=(original_h, original_w), 
            mode="bilinear", 
            align_corners=False
        )
        
        # 6. Get the predicted class for each pixel
        predictions = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # 7. Create a binary mask for the sky
        sky_mask = (predictions == SKY_CLASS_ID).astype(np.uint8) * 255

        # Convert BGR to Grayscale for overall luminance and glare detection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Convert BGR to Grayscale for overall luminance
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # ==========================================
        # 1. GLOBAL NIGHT BYPASS
        # ==========================================
        global_mean = np.mean(gray) / 255.0
        global_p90 = np.percentile(gray, 90) / 255.0

        # If the whole image is fundamentally dark, ignore SegFormer's sky mask.
        if global_mean < 0.15 and global_p90 < 0.50:
            time_of_day = "night"
            cloud_fraction = 0.0
            mean_sky = 0.0
            p90_sky = 0.0
        else:
            # --- DAY/DUSK CLOUD & LIGHTING LOGIC ---
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            S = hsv[:, :, 1]
            V = hsv[:, :, 2]
            
            sky_pixels_bool = (sky_mask == 255)
            total_sky_pixels = np.sum(sky_pixels_bool)
            
            if total_sky_pixels > 0:
                cloud_pixels_mask = sky_pixels_bool & (S < 80) & (V > 100)
                cloud_fraction = float(np.sum(cloud_pixels_mask) / total_sky_pixels)
                
                actual_sky_pixels = gray[sky_pixels_bool]
                mean_sky = np.mean(actual_sky_pixels) / 255.0
                p90_sky = np.percentile(actual_sky_pixels, 90) / 255.0
                
                # We already know it's not strictly night, so classify Day vs Dusk
                if mean_sky > 0.40 or p90_sky > 0.80:
                    time_of_day = "day"
                else:
                    time_of_day = "dusk/dawn"
            else:
                cloud_fraction = 0.0
                time_of_day = "unknown (no sky)"
                mean_sky = 0.0
                p90_sky = 0.0

        # ==========================================
        # 2. SMART GLARE DETECTION (Blob Filtering)
        # ==========================================
        # Blur heavily to merge scattered bright spots
        blurred_gray = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Threshold for extreme brightness
        _, overexposed_mask = cv2.threshold(blurred_gray, 245, 255, cv2.THRESH_BINARY)
        
        # Exclude the sky so we don't flag the normal sun
        non_sky_mask = cv2.bitwise_not(sky_mask)
        potential_glare = cv2.bitwise_and(overexposed_mask, overexposed_mask, mask=non_sky_mask)
        
        # Exclude the bottom 20% of the image (ignores ego-car hood reflections)
        cutoff_y = int(original_h * 0.8)
        potential_glare[cutoff_y:, :] = 0

        # Find contiguous blobs of brightness
        contours, _ = cv2.findContours(potential_glare, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dangerous_glare_mask = np.zeros_like(gray)
        total_image_area = original_h * original_w
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # A blob must be larger than 1.5% of the total image to be considered "blinding"
            # This ignores white cars, streetlights, and painted lane lines
            if area > (total_image_area * 0.015):
                cv2.drawContours(dangerous_glare_mask, [contour], -1, 255, -1)

        glare_pixel_count = np.count_nonzero(dangerous_glare_mask)
        glare_ratio = float(glare_pixel_count / total_image_area)
        
        has_severe_glare = bool(glare_ratio > 0.0) # If any blobs survived the size filter, flag it
        # ==========================================

        # Save Combined Metrics to JSON
        json_filename = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(json_dir, json_filename)
        
        results_dict = {
            "cloud_fraction": round(cloud_fraction, 4),
            "lighting_condition": time_of_day,
            "light_level_p90": round(float(p90_sky), 3),
            "glare": {
                "has_severe_glare": has_severe_glare,
                "glare_ratio": round(glare_ratio, 4)
            },
            "debug_metrics": {
                "sky_mean": round(float(mean_sky), 3),
                "sky_p90": round(float(p90_sky), 3)
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)

        # --- VISUALIZATION OVERLAYS ---
        # Optional: Add a black mask to the sky mask to hide all non-sky pixels in box
        overlay = img_bgr.copy()
        overlay[sky_mask != 255] = [0, 0, 0]
        cv2.addWeighted(overlay, 0.3, img_bgr, 0.7, 0, img_bgr)

        # Find contours of the sky mask to draw bounding boxes
        contours, _ = cv2.findContours(sky_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img_bgr, f"Sky: {time_of_day.upper()}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Find contours of the dangerous glare and draw them in Yellow
        glare_contours, _ = cv2.findContours(dangerous_glare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for gc in glare_contours:
            if cv2.contourArea(gc) > 50: # Slightly lower threshold for 480x270 images
                cv2.drawContours(img_bgr, [gc], -1, (0, 255, 255), 1) 
                
        # Add a warning tag if glare is detected
        if has_severe_glare:
            cv2.putText(img_bgr, "WARNING: SEVERE GLARE", (10, original_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 9. Save the augmented image
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, img_bgr)
        glare_log = " | GLARE WARNING" if has_severe_glare else ""
        print(f"Saved: {out_path} -> {time_of_day.upper()} (P90: {round(p90_sky, 2)}){glare_log}")

if __name__ == "__main__":
    process_images(input_dir="../../source_images", output_dir="output_boxes", json_dir="output_json")