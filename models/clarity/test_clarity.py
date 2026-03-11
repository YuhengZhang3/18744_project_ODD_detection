import os
import cv2
import glob
import numpy as np

def calculate_clarity_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    safe_sharpness = max(sharpness, 1.0) 
    clarity_score = (contrast * np.log10(safe_sharpness)) / 10.0
    return clarity_score

def evaluate_clarity(input_dir="../../source_images", output_dir="clarity_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        print(f"No images found in '{input_dir}'.")
        return
        
    print(f"Found {len(image_paths)} images. Starting Pass 1: Calculating scores...")

    # --- PASS 1: Calculate all scores without storing images in memory ---
    scored_images = []
    for img_path in image_paths:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
            
        score = calculate_clarity_score(img_bgr)
        scored_images.append((score, img_path))
        
    print("Sorting images by clarity score...")
    
    # --- SORT: Lowest clarity (worst) to Highest clarity (best) ---
    scored_images.sort(key=lambda x: x[0])

    print("Starting Pass 2: Drawing overlays and saving with indexed names...")

    # --- PASS 2: Re-load, draw text, and save with the sorted index ---
    for index, (score, img_path) in enumerate(scored_images):
        img_bgr = cv2.imread(img_path)
        h, w = img_bgr.shape[:2]

        result = img_bgr.copy()
        
        # --- SCALED BACKGROUND BOX ---
        scale_factor = max(0.4, h / 720.0) 
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE_MAIN = 0.8 * scale_factor
        FONT_SCALE_SUB = 0.6 * scale_factor
        THICK = max(1, int(2 * scale_factor))
        LINE_HEIGHT = max(20, int(35 * scale_factor))
        
        lines = [
            (f"Score: {score:.1f}", (255, 255, 255), FONT_SCALE_SUB)
        ]

        # Calculate exact box dimensions based on text length
        max_width = 0
        for text, text_color, scale in lines:
            (tw, th), _ = cv2.getTextSize(text, FONT, scale, THICK)
            if tw > max_width: max_width = tw
            
        box_width = max_width + int(20 * scale_factor)
        box_height = int(len(lines) * LINE_HEIGHT + 15 * scale_factor)
        
        # Draw semi-transparent background
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result) # 60% opacity

        # Draw the text
        current_y = int(30 * scale_factor)
        x_pad = int(10 * scale_factor)

        for text, text_color, scale in lines:
            cv2.putText(result, text, (x_pad, current_y), FONT, scale, text_color, THICK, cv2.LINE_AA)
            current_y += LINE_HEIGHT

        # --- SAVE USING THE INDEX ---
        # Get original extension (e.g., .jpg) to keep filetypes intact
        ext = os.path.splitext(img_path)[1]
        
        # Zero-pad the index so "2" becomes "0002" (keeps file explorer sorting intact)
        new_filename = f"{index:04d}{ext}" 
        output_filename = os.path.join(output_dir, new_filename)
        
        cv2.imwrite(output_filename, result)
        print(f"Saved {new_filename} (Score: {score:.1f})")

    print(f"\nDone! Your images are sorted worst-to-best in '{output_dir}'.")

if __name__ == "__main__":
    evaluate_clarity()