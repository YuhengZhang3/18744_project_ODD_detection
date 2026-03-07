import os
import cv2
import glob
import torch
import numpy as np
from transformers import SegformerForSemanticSegmentation

def evaluate_test_set(model_dir="custom_glare_model", test_dir="../../source_images", output_dir="glare_predictions"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading custom model from '{model_dir}' onto {str(device).upper()}...")

    # Create output directory for the results
    os.makedirs(output_dir, exist_ok=True)

    # Load our newly trained model
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(device)
    model.eval()

    # Find all images in your usual test directory (handles jpg, jpeg, and png)
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(test_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(test_dir, ext.upper()))) # Catch uppercase extensions

    if not image_paths:
        print(f"No images found in '{test_dir}/'. Please check the folder path.")
        return
        
    print(f"Found {len(image_paths)} images in '{test_dir}'. Running inference...")

    for img_path in image_paths:
        # 1. Load and prep image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]
        
        # Resize to 512x512 (what the model was trained on)
        img_resized = cv2.resize(img_rgb, (512, 512))
        img_tensor = torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)

        # 2. Run Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            
        # 3. Upsample the low-res logits to match the original image size
        logits = torch.nn.functional.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
        
        # 4. Get the predicted class (0 = Background, 1 = Glare)
        predicted_mask = logits.argmax(dim=1)[0].cpu().numpy()

        # 5. Draw the detection regions
        overlay = img_bgr.copy()
        # Mark predicted glare pixels with bright yellow [0, 255, 255] in BGR
        overlay[predicted_mask == 1] = [0, 255, 255] 
        
        # Blend the original image and the yellow overlay (70% original, 30% yellow)
        result = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)

        # 6. Save the result
        filename = os.path.basename(img_path)
        output_filename = os.path.join(output_dir, f"pred_{filename}")
        cv2.imwrite(output_filename, result)
        print(f"Processed and saved: {output_filename}")

    print(f"\nDone! Check the '{output_dir}' folder to review the glare detection regions.")

if __name__ == "__main__":
    evaluate_test_set()