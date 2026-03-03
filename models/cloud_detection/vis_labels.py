import cv2
import os
import json
import glob

def visualize_metrics(image_dir="output_boxes", json_dir="output_json"):
    # 1. Grab all augmented images
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.[jp][pn]*[g]")))
    
    if not image_paths:
        print(f"No images found in {image_dir}. Did you run the segmentation script?")
        return

    print(f"Found {len(image_paths)} images to review.")
    
    idx = 0
    while idx < len(image_paths):
        img_path = image_paths[idx]
        filename = os.path.basename(img_path)
        
        # 2. Match the image with its JSON file
        json_path = os.path.join(json_dir, os.path.splitext(filename)[0] + '.json')

        img = cv2.imread(img_path)
        if img is None:
            idx += 1
            continue

        # --- Scale to 720p ---
        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        display_img = img.copy()

        # 3. Read ALL metrics
        cloud_text = "N/A"
        condition_text = "UNKNOWN"
        p90_score_text = "N/A"
        sky_mean_text = "N/A"
        sky_p90_text = "N/A"
        glare_text = "N/A"
        
        cf_val = 0.0
        severe_glare = False
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                    # Cloud metrics
                    cf_val = data.get("cloud_fraction", 0.0)
                    cloud_text = f"{cf_val * 100:.1f}%"
                    
                    # Lighting metrics
                    condition_text = str(data.get("lighting_condition", "unknown")).upper()
                    p90_score_text = str(data.get("light_level_p90", "N/A"))
                    
                    # Glare metrics
                    glare_data = data.get("glare", {})
                    severe_glare = glare_data.get("has_severe_glare", False)
                    glare_ratio = glare_data.get("glare_ratio", 0.0)
                    glare_text = f"{glare_ratio * 100:.1f}%"

                    # Debug metrics
                    debug = data.get("debug_metrics", {})
                    sky_mean_text = str(debug.get("sky_mean", "N/A"))
                    sky_p90_text = str(debug.get("sky_p90", "N/A"))
            except Exception as e:
                condition_text = "JSON ERROR"

        # --- UI Overlay (Scaled for 1280x720) ---
        
        # Header
        header_height = 80
        overlay = display_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, display_img, 0.15, 0, display_img)

        # 1. Draw Title
        title_text = f"FILE: {filename}"
        cv2.putText(display_img, title_text, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # 2. Draw Lighting Condition (Color Coded)
        cond_color = (255, 255, 255) # White default
        if condition_text == "DAY": cond_color = (200, 255, 255)       
        elif condition_text == "DUSK/DAWN": cond_color = (0, 150, 255) 
        elif condition_text == "NIGHT": cond_color = (255, 100, 100)   
        elif "UNKNOWN" in condition_text: cond_color = (100, 100, 100) 
        
        cv2.putText(display_img, f"COND: {condition_text}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cond_color, 2)
        cv2.putText(display_img, f"P90: {p90_score_text}", (230, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 3. Draw Cloud Cover
        cloud_color = (255, 255, 255) if cf_val > 0.3 else (255, 200, 50)
        cv2.putText(display_img, f"CLOUD: {cloud_text}", (420, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cloud_color, 2)

        # 4. Draw Glare Metrics
        glare_color = (0, 0, 255) if severe_glare else (150, 255, 150) # Red if dangerous, Green if safe
        cv2.putText(display_img, f"GLARE: {glare_text}", (640, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, glare_color, 2)

        # 5. Draw Debug Metrics
        cv2.putText(display_img, f"MEAN: {sky_mean_text}", (w - 250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1)
        cv2.putText(display_img, f"SKY P90: {sky_p90_text}", (w - 250, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # Footer for controls
        cv2.rectangle(display_img, (0, h - 35), (w, h), (10, 10, 10), -1)
        footer_text = f"[{idx+1}/{len(image_paths)}] Controls: [N] Next | [B] Back | [Q] Quit"
        cv2.putText(display_img, footer_text, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # --- Display ---
        cv2.imshow("Multi-Metric Visualizer - 720p", display_img)
        
        # --- Keyboard Navigation ---
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            idx += 1
        elif key == ord('b'):
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_metrics(image_dir="output_boxes", json_dir="output_json")