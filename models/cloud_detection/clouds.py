import os
import cv2
import json
import torch
import numpy as np
import glob
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from concurrent.futures import ThreadPoolExecutor

## NO TRAINING NEEDED


def _load_single_image(img_path):
    """Load one image from disk. Runs in a thread pool (cv2 releases the GIL)."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    filename = os.path.basename(img_path)
    h, w = img_bgr.shape[:2]
    return (img_rgb, img_bgr, filename, h, w)
 
 
def process_clouds(input_dir="../../source_images", output_dir="output_boxes", json_dir="output_json",
                   batch_size=16, save_vis=False, num_workers=6):
    # 1. Setup directories
    os.makedirs(json_dir, exist_ok=True)
    if save_vis:
        os.makedirs(output_dir, exist_ok=True)
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
    print(f"Processing {len(image_paths)} images for cloud coverage (batch={batch_size}, workers={num_workers})...")
 
    # 3. Process in batches with prefetched image loading
    executor = ThreadPoolExecutor(max_workers=num_workers)
 
    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start:batch_start + batch_size]
        
        # --- Parallel image loading ---
        futures = [executor.submit(_load_single_image, p) for p in batch_paths]
        loaded = [f.result() for f in futures]
        loaded = [x for x in loaded if x is not None]
        
        if not loaded:
            continue
        
        batch_rgb = [x[0] for x in loaded]
        batch_bgr = [x[1] for x in loaded]
        batch_filenames = [x[2] for x in loaded]
        batch_sizes = [(x[3], x[4]) for x in loaded]
        
        # --- GPU: Batched SegFormer inference ---
        inputs = processor(images=batch_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits  # [B, num_classes, H_small, W_small]
        
        # --- CPU: Per-image post-processing ---
        for i in range(len(batch_rgb)):
            original_h, original_w = batch_sizes[i]
            filename = batch_filenames[i]
            img_bgr = batch_bgr[i]
            
            # Upsample to original size (per-image, since sizes differ)
            single_logit = logits[i:i+1]
            upsampled = nn.functional.interpolate(
                single_logit,
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=False
            )
            predictions = upsampled.argmax(dim=1)[0].cpu().numpy()
            sky_mask = (predictions == SKY_CLASS_ID).astype(np.uint8) * 255
 
            # --- CLOUD DETECTION LOGIC ---
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            S = hsv[:, :, 1]
            V = hsv[:, :, 2]
            
            sky_pixels_bool = (sky_mask == 255)
            total_sky_pixels = np.sum(sky_pixels_bool)
            
            cloud_fraction = 0.0
            if total_sky_pixels > 0:
                cloud_pixels_bool = sky_pixels_bool & (S < 80) & (V > 100)
                cloud_fraction = float(np.sum(cloud_pixels_bool) / total_sky_pixels)
 
            # --- SAVE METRICS TO JSON ---
            json_filename = os.path.splitext(filename)[0] + '.json'
            json_path = os.path.join(json_dir, json_filename)
            with open(json_path, 'w') as f:
                json.dump({"cloud_fraction": round(cloud_fraction, 4)}, f, indent=4)
 
            # --- OPTIONAL VISUALIZATION ---
            if save_vis:
                overlay = img_bgr.copy()
                overlay[sky_mask != 255] = [0, 0, 0]
                cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
 
                contours, _ = cv2.findContours(sky_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < 500:
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
                out_path = os.path.join(output_dir, filename)
                cv2.imwrite(out_path, img_bgr)
 
        processed = min(batch_start + batch_size, len(image_paths))
        print(f"  Cloud detection: {processed}/{len(image_paths)} images processed")
 
    executor.shutdown(wait=False)
 
if __name__ == "__main__":
    process_clouds(input_dir="../../source_images", output_dir="output_boxes", json_dir="output_json")