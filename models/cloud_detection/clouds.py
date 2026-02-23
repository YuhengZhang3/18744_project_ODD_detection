import os
import cv2
import torch
import numpy as np
import glob
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

def process_images(input_dir="input_images", output_dir="output_images"):
    # 1. Setup directories
    os.makedirs(output_dir, exist_ok=True)
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

        # Optional: Add a light blue overlay to the sky mask so you can see the segmentation
        overlay = img_bgr.copy()
        overlay[sky_mask == 255] = [255, 200, 100] # Light blue in BGR
        cv2.addWeighted(overlay, 0.3, img_bgr, 0.7, 0, img_bgr)

        # 8. Find contours of the sky mask to draw bounding boxes
        contours, _ = cv2.findContours(sky_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Ignore tiny patches of "sky" (noise)
            if cv2.contourArea(contour) < 500:
                continue
                
            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw the bounding box (Red color, thickness 2)
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img_bgr, "Sky", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 9. Save the augmented image
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, img_bgr)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    process_images(input_dir="../../source_images", output_dir="output_images")