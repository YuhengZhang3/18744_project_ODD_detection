import cv2
import os
import json
import glob

def visualize_weather(image_dir="../../source_images", predictions_file="predictions.json"):
    # Load predictions
    with open(predictions_file, "r") as f:
        predictions = json.load(f)

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

        img = cv2.imread(img_path)
        if img is None:
            idx += 1
            continue

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
        if filename in predictions:
            pred = predictions[filename]
            pred_text = (
                f"Fog: {pred['fog_severity']:.2f} | "
                f"Rain: {pred['rain_severity']:.2f} | "
                f"Snow: {pred['snow_severity']:.2f}"
            )
        else:
            pred_text = "Predictions Missing"

        # Draw prediction text
        cv2.putText(display_img, pred_text, (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
        elif key == ord('b'):
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_weather(image_dir="../../source_images", predictions_file="predictions.json")