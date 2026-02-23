import cv2
import os
import json
import glob

def visualize_cloudiness(image_dir="output_boxes", json_dir="output_json"):
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

        h, w = img.shape[:2]
        display_img = img.copy()

        # 3. Read the cloud fraction metric
        cloud_text = "JSON Missing"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    cf = data.get("cloud_fraction", 0.0)
                    # Convert 0.0-1.0 to a clean percentage string
                    cloud_text = f"{cf * 100:.1f}%"
            except Exception as e:
                cloud_text = "JSON Error"

        # --- UI Overlay ---
        
        # Create a semi-transparent black header bar at the top for readability
        header_height = 60
        overlay = display_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), (20, 20, 20), -1)
        # 75% opacity for the header
        cv2.addWeighted(overlay, 0.75, display_img, 0.25, 0, display_img)

        # Draw Title and Metric in the header
        title_text = f"FILE: {filename}"
        cv2.putText(display_img, title_text, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        metric_text = f"CLOUD COVER: {cloud_text}"
        # Highlight high cloud cover in white, low cloud cover in blue
        color = (255, 255, 255) if cloud_text != "JSON Missing" and float(cf) > 0.3 else (255, 200, 50)
        cv2.putText(display_img, metric_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Create a solid black footer for controls
        cv2.rectangle(display_img, (0, h - 30), (w, h), (10, 10, 10), -1)
        footer_text = f"[{idx+1}/{len(image_paths)}] Controls: [N] Next | [B] Back | [Q] Quit"
        cv2.putText(display_img, footer_text, (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # --- Display ---
        cv2.imshow("Cloud Cover Visualizer", display_img)
        
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
    # Point these to the directories created by the sky_segmentation.py script
    visualize_cloudiness(image_dir="output_boxes", json_dir="output_json")