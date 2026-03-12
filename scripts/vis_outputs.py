import os
import cv2
import json
import glob
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

IMAGE_DIR = os.path.join(project_root, "source_images")
JSON_ROOT = os.path.join(project_root, "outputs")

GLARE_DIR = os.path.join(JSON_ROOT, "glare")
YUHENG_DIR = os.path.join(JSON_ROOT, "yuheng")


def load_json_data(subfolder, basename):
    json_path = os.path.join(JSON_ROOT, subfolder, f"{basename}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON"}
    return None


def load_unified_prediction(basename):
    json_path = os.path.join(YUHENG_DIR, f"{basename}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None


def draw_hud(img, text_lines):

    if not text_lines:
        return img

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_spacing = 25
    margin = 15

    max_text_width = max(
        [cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in text_lines]
    )

    box_width = max_text_width + (margin * 2)
    box_height = (len(text_lines) * line_spacing) + (margin * 2)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), -1)

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    y_offset = margin + 15

    for line in text_lines:
        cv2.putText(img, line, (margin, y_offset), font, font_scale, (255, 255, 255), thickness)
        y_offset += line_spacing

    return img


def main():

    image_paths = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext.upper())))

    image_paths.sort()

    if not image_paths:
        print(f"No images found in {IMAGE_DIR}")
        return

    print("========================================")
    print("        PIPELINE VIEWER STARTED         ")
    print("========================================")

    idx = 0

    while True:

        img_path = image_paths[idx]
        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]

        img = cv2.imread(img_path)

        if img is None:
            idx = (idx + 1) % len(image_paths)
            continue

        # -------- Glare Overlay --------
        glare_mask_path = os.path.join(GLARE_DIR, f"{basename}.png")

        if os.path.exists(glare_mask_path):

            mask = cv2.imread(glare_mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is not None:

                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                overlay = img.copy()
                overlay[mask >= 128] = [0, 255, 255]

                cv2.addWeighted(img, 0.7, overlay, 0.3, 0, img)

        # -------- Resize for display --------
        max_display_height = 800

        if img.shape[0] > max_display_height:
            scale = max_display_height / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

        # -------- Load pipeline outputs --------
        weather_data = load_json_data("weather", basename)
        cloud_data = load_json_data("cloud_detection", basename)
        unified = load_unified_prediction(basename)

        hud_lines = [f"FILE: {filename}", "-" * 30]

        # -------- Weather --------
        if weather_data and "weather" in weather_data:
            w = weather_data["weather"]

            hud_lines.append(
                f"WEATHER: Fog({w.get('fog_severity',0):.2f}) "
                f"Rain({w.get('rain_severity',0):.2f}) "
                f"Snow({w.get('snow_severity',0):.2f})"
            )
        else:
            hud_lines.append("WEATHER: No Data")

        # -------- Unified ODD model --------
        if unified and "predictions" in unified:

            p = unified["predictions"]

            time = p["time"]["label"]
            time_conf = p["time"]["confidence"]

            scene = p["scene"]["label"]
            scene_conf = p["scene"]["confidence"]

            vis = p["visibility"]["label"]
            vis_conf = p["visibility"]["confidence"]

            road = p["road_condition"]["label"]
            road_conf = p["road_condition"]["confidence"]

            hud_lines.append(f"TIME: {time} ({time_conf:.2f})")
            hud_lines.append(f"SCENE: {scene} ({scene_conf:.2f})")
            hud_lines.append(f"VISIBILITY: {vis} ({vis_conf:.2f})")
            hud_lines.append(f"ROAD: {road} ({road_conf:.2f})")

        else:
            hud_lines.append("ODD MODEL: No Data")

        # -------- Clouds --------
        if cloud_data and "cloud_fraction" in cloud_data:
            hud_lines.append(f"CLOUDS: Fraction {cloud_data['cloud_fraction']:.3f}")
        else:
            hud_lines.append("CLOUDS: No Data")

        hud_lines.append("GLARE: See overlay")

        display_img = draw_hud(img, hud_lines)

        cv2.imshow("Vision Pipeline Viewer", display_img)

        key = cv2.waitKey(0) & 0xFF

        if key in [ord('q'), 27]:
            break
        elif key in [ord('d'), 83]:
            idx = (idx + 1) % len(image_paths)
        elif key in [ord('a'), 81]:
            idx = (idx - 1) % len(image_paths)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()