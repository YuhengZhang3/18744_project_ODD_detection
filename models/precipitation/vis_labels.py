import cv2
import os
import json
import glob

def visualize_weather(image_dir="../../source_images", json_dir="output_json"):
    # 1. Grab all raw images
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.[jp][pn]*[g]")))
    
    if not image_paths:
        print(f"No images found in {image_dir}.")
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

        h, w = img.shape[:2]
        display_img = img.copy()

        # 3. Read the weather metrics
        precip_text = "JSON Missing"
        severity_text = "N/A"
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    precip_text = str(data.get("precipitation", "unknown")).upper()
                    severity_text = str(data.get("severity", 0.0))
            except Exception as e:
                precip_text = "JSON Error"

        # --- UI Overlay ---
        # Create a semi-transparent black header bar at the top
        header_height = 80
        overlay = display_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, display_img, 0.15, 0, display_img)

        # Draw Title
        title_text = f"FILE: {filename}"
        cv2.putText(display_img, title_text, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Color-code based on precipitation type
        color = (255, 255, 255)
        if precip_text == "RAIN": color = (255, 150, 0)  # Blue-ish
        elif precip_text == "NONE": color = (100, 255, 100) # Green
        elif precip_text == "SNOW": color = (255, 255, 255) # White
        
        # Draw Metrics
        cv2.putText(display_img, f"PRECIPITATION: {precip_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_img, f"SEVERITY: {severity_text}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Create a solid black footer for controls
        cv2.rectangle(display_img, (0, h - 30), (w, h), (10, 10, 10), -1)
        footer_text = f"[{idx+1}/{len(image_paths)}] Controls: [n] Next | [b] Back | [q] Quit"
        cv2.putText(display_img, footer_text, (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # --- Display ---
        cv2.imshow("Weather Data Auditor", display_img)
        
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
    visualize_weather(image_dir="../../source_images", json_dir="output_json")