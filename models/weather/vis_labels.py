import cv2
import os
import json
import glob

def visualize_weather(image_dir="../../source_images", json_dir="output_json"):
    # Grab all images in the folder
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.[jp][pn]*[g]")))

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
        pred = None
        
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    # Extract the metrics from the "weather" key we defined earlier
                    pred = data.get("weather", None)
            except json.JSONDecodeError:
                print(f"Error reading JSON for {filename}")

        h, w = img.shape[:2]
        display_img = img.copy()

        # --- Overlay Header ---
        overlay = display_img.copy()
        header_height = 80
        cv2.rectangle(overlay, (0, 0), (w, header_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, display_img, 0.25, 0, display_img)

        # Draw filename
        cv2.putText(display_img, f"FILE: {filename}", (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Draw predictions
        if pred:
            pred_text = (
                f"Fog: {pred.get('fog_severity', 0.0):.2f} | "
                f"Rain: {pred.get('rain_severity', 0.0):.2f} | "
                f"Snow: {pred.get('snow_severity', 0.0):.2f}"
            )
            color = (0, 255, 0) # Green for successful load
        else:
            pred_text = "Predictions Missing"
            color = (0, 0, 255) # Red for missing JSON

        # Draw prediction text
        cv2.putText(display_img, pred_text, (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- Overlay Footer ---
        cv2.rectangle(display_img, (0, h - 30), (w, h), (10, 10, 10), -1)
        footer_text = f"[{idx+1}/{len(image_paths)}] Controls: [N] Next | [B] Back | [Q] Quit"
        cv2.putText(display_img, footer_text, (20, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # --- Display ---
        cv2.imshow("Weather Prediction Visualizer", display_img)

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
    # Updated to point to the output_json directory
    visualize_weather(image_dir="../../source_images", json_dir="output_json")