import os
import cv2
import glob
import json
import torch
import numpy as np
from transformers import SegformerForSemanticSegmentation
from concurrent.futures import ThreadPoolExecutor
 
 
def _load_and_resize(img_path):
    """Load one image and resize to 512x512. Runs in thread pool (cv2 releases GIL)."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    img_resized = cv2.resize(img_rgb, (512, 512))
    tensor = torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    filename = os.path.basename(img_path)
    basename = os.path.splitext(filename)[0]
    return (tensor, basename, h, w)
 
 
def evaluate_test_set(model_dir="custom_glare_model", test_dir="../../source_images", output_dir="glare_predictions",
                      batch_size=32, save_mask=False, num_workers=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading custom model from '{model_dir}' onto {str(device).upper()}...")
 
    os.makedirs(output_dir, exist_ok=True)
 
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(device)
    model.eval()
 
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(test_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(test_dir, ext.upper())))
 
    if not image_paths:
        print(f"No images found in '{test_dir}/'. Please check the folder path.")
        return
        
    print(f"Found {len(image_paths)} images in '{test_dir}'. Running inference (batch={batch_size}, workers={num_workers})...")
 
    executor = ThreadPoolExecutor(max_workers=num_workers)
 
    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start:batch_start + batch_size]
        
        # --- Parallel image loading + resize ---
        futures = [executor.submit(_load_and_resize, p) for p in batch_paths]
        loaded = [f.result() for f in futures]
        loaded = [x for x in loaded if x is not None]
        
        if not loaded:
            continue
        
        batch_tensors = [x[0] for x in loaded]
        batch_basenames = [x[1] for x in loaded]
        batch_sizes = [(x[2], x[3]) for x in loaded]
        
        # --- GPU: Batched inference ---
        batch_input = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            outputs = model(batch_input)
        
        logits = outputs.logits  # [B, 2, H_small, W_small]
        
        # --- CPU: Per-image post-processing ---
        for i, (basename, (orig_h, orig_w)) in enumerate(zip(batch_basenames, batch_sizes)):
            single_logit = logits[i:i+1]
            upsampled = torch.nn.functional.interpolate(
                single_logit, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )
            predicted_mask = upsampled.argmax(dim=1)[0].cpu().numpy()
            
            # Compute glare_ratio
            total_pixels = orig_h * orig_w
            glare_pixels = int(predicted_mask.sum())
            glare_ratio = round(glare_pixels / total_pixels, 4) if total_pixels > 0 else 0.0
            
            # Save JSON (always)
            json_path = os.path.join(output_dir, f"{basename}.json")
            with open(json_path, 'w') as f:
                json.dump({"glare_ratio": glare_ratio}, f, indent=4)
            
            # Save mask PNG (optional)
            if save_mask:
                mask_img = (predicted_mask * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, f"{basename}.png"), mask_img)
 
        processed = min(batch_start + batch_size, len(image_paths))
        print(f"  Glare detection: {processed}/{len(image_paths)} images processed")
 
    executor.shutdown(wait=False)
    print(f"\nDone! Glare results saved to '{output_dir}'")
 
if __name__ == "__main__":
    evaluate_test_set(save_mask=True)