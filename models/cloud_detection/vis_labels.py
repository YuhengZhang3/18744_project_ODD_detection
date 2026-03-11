import cv2
import os
import json
import glob

def visualize_clouds(image_dir="output_boxes", json_dir="output_json"):
    # 1. Grab all augmented images
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.[jp][pn]*[g]")))
    
    if not image_paths:
        print(f"No images found in '{image_dir}'. Did you run the detection script?")
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

        # --- Scale to 720p for consistent viewing ---
        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        display_img = img.copy()

        # 3. Read Cloud Metric from JSON
        cf_val = 0.0
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    cf_val = data.get("cloud_fraction", 0.0)
            except Exception as e:
                print(f"Error reading JSON for {filename}: {e}")

        cloud_text = f"{cf_val * 100:.1f}%"

        # --- UI OVERLAYS ---
        
        # Top Header (Solid background for readability)
        cv2.rectangle(display_img, (0, 0), (w, 50), (20, 20, 20), -1)
        
        # 1. Draw Title
        cv2.putText(display_img, f"FILE: {filename}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # 2. Draw Cloud Cover (Yellow if cover is low, White if cloudy)
        cloud_color = (255, 255, 255) if cf_val > 0.3 else (50, 200, 255)
        cv2.putText(display_img, f"JSON CLOUD COVER: {cloud_text}", (w - 350, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cloud_color, 2)

        # Footer for controls
        cv2.rectangle(display_img, (0, h - 40), (w, h), (10, 10, 10), -1)
        footer_text = f"[{idx+1}/{len(image_paths)}] Controls: [N] Next | [B] Back | [Q] Quit"
        cv2.putText(display_img, footer_text, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # --- DISPLAY ---
        cv2.imshow("Cloud Cover Visualizer", display_img)
        
        # --- KEYBOARD NAVIGATION ---
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            idx += 1
        elif key == ord('b'):
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_clouds(image_dir="output_boxes", json_dir="output_json")