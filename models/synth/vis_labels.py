import cv2
import os
import json
import glob

def visualize_sensors(image_dir="../../source_images", json_dir="output_json"):
    # Grab all images in the folder safely
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
    # Sort them so navigation is consistent
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images to review.")

    idx = 0
    while idx < len(image_paths):
        img_path = image_paths[idx]
        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]

        img = cv2.imread(img_path)
        if img is None:
            idx += 1
            continue

        # --- Load the corresponding JSON file for this image ---
        json_path = os.path.join(json_dir, f"{basename}.json")
        sensor_data = None
        
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    # Extract the metrics from the "sensors" key
                    sensor_data = data.get("sensors", None)
            except json.JSONDecodeError:
                print(f"Error reading JSON for {filename}")

        h, w = img.shape[:2]
        display_img = img.copy()

        # --- Overlay Header ---
        overlay = display_img.copy()
        # Increased header height slightly to fit the new city row
        header_height = 135 
        cv2.rectangle(overlay, (0, 0), (w, header_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, display_img, 0.15, 0, display_img)

        # Draw filename
        cv2.putText(display_img, f"FILE: {filename}", (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Draw predictions
        if sensor_data:
            # 1. Base CLIP environment data
            line1_text = (
                f"Time: {sensor_data.get('clock_time', 'N/A')} | "
                f"Temp: {sensor_data.get('temperature_c', 'N/A')} C | "
                f"Hum: {sensor_data.get('humidity_pct', 'N/A')}%"
            )
            
            # 2. New GeoCLIP location data
            location = sensor_data.get("location", {})
            lat = location.get("lat", "N/A")
            lon = location.get("lon", "N/A")
            conf = location.get("geoclip_confidence", "N/A")
            city = location.get("nearest_city", "Unknown Location")
            
            line2_text = f"GeoCLIP GPS: {lat}, {lon} | Conf: {conf}"
            line3_text = f"Predicted City: {city}"
            
            # Draw the three rows of sensor data
            cv2.putText(display_img, line1_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2) # Green
            cv2.putText(display_img, line2_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2) # Cyan
            cv2.putText(display_img, line3_text, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 100), 2) # Light Blue
            
        else:
            cv2.putText(display_img, "Sensor Predictions Missing", (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- Overlay Footer ---
        cv2.rectangle(display_img, (0, h - 30), (w, h), (10, 10, 10), -1)
        footer_text = f"[{idx+1}/{len(image_paths)}] Controls: [N] Next | [B] Back | [Q] Quit"
        cv2.putText(display_img, footer_text, (20, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # --- Display ---
        cv2.imshow("Multimodal Sensor Visualizer", display_img)

        # --- Keyboard Navigation ---
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            idx += 1
        elif key == ord('m'):
            idx += 1000
        elif key == ord('b'):
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_sensors(image_dir="../../source_images", json_dir="output_json")