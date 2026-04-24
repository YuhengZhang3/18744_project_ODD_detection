import os
import cv2
import json
import glob
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

IMAGE_DIR = os.path.join(project_root, "outputs/yolo_vis")
JSON_ROOT = os.path.join(project_root, "outputs")

MERGED_DIR = os.path.join(JSON_ROOT, "merged_json")
GLARE_DIR = os.path.join(JSON_ROOT, "glare")
OUTPUT_DIR = os.path.join(JSON_ROOT, "visualize_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_merged_data(basename):
    json_path = os.path.join(MERGED_DIR, f"{basename}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def draw_side_panel(img, text_lines, panel_width=640):
    h, w = img.shape[:2]

    canvas = np.zeros((h, w + panel_width, 3), dtype=np.uint8)

    canvas[:, :w] = img
    canvas[:, w:] = (30, 30, 30)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_spacing = 22

    x = w + 15
    y = 30

    for line in text_lines:
        cv2.putText(canvas, line, (x, y),
                    font, font_scale, (255, 255, 255), thickness)
        y += line_spacing

    return canvas


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
    print("        PIPELINE EXPORTER STARTED       ")
    print("========================================")

    for img_path in image_paths:

        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]

        img = cv2.imread(img_path)

        if img is None:
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

        # -------- Resize --------
        def resize_with_padding(img, target_size=(1280, 720)):
            target_w, target_h = target_size
            h, w = img.shape[:2]

            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)

            resized = cv2.resize(img, (new_w, new_h))

            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2

            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            return canvas

        img = resize_with_padding(img, (1280, 720))

        # -------- YOLO bbox toggle (always off in exporter) --------
        # (kept logic but no toggle loop)
        show_bbox = False
        if show_bbox:
            bbox_path = os.path.join(JSON_ROOT, "yolo_vis", f"{basename}.jpg")
            if os.path.exists(bbox_path):
                bbox_img = cv2.imread(bbox_path)
                if bbox_img is not None:
                    bbox_img = cv2.resize(bbox_img, (img.shape[1], img.shape[0]))
                    img = bbox_img

        # -------- Load Data --------
        merged_data = load_merged_data(basename)

        synth_data = merged_data.get("synth_outputs") or {}
        weather_data = merged_data.get("weather") or {}
        unified = merged_data.get("yuheng") or {}
        stage2 = merged_data.get("stage2") or {}
        yolo_data = merged_data.get("yolo") or {}
        cloud_data = merged_data.get("cloud_detection") or {}
        glare_data = merged_data.get("glare") or {}

        hud_lines = [f"FILE: {filename}", "-" * 40]

        # -------- Sensors --------
        if "sensors" in synth_data:
            s = synth_data["sensors"]

            hud_lines.append(
                f"SENSORS: {s.get('clock_time','N/A')} | "
                f"{s.get('temperature_c','N/A')}C | "
                f"{s.get('humidity_pct','N/A')}%"
            )

            loc = s.get("location", {})
            if loc:
                hud_lines.append(f"LOC: {loc.get('nearest_city','Unknown')}")
        else:
            hud_lines.append("SENSORS: No Data")

        # -------- Weather --------
        w = weather_data.get("weather", weather_data) if weather_data else {}
        wc = stage2.get("weather_corrected", {})

        def fmt_weather(orig_key, corr_key):
            o = w.get(orig_key, None)
            c = wc.get(corr_key, None)

            if o is None and c is None:
                return f"{orig_key.split('_')[0].upper()}: N/A"

            if o is not None and c is not None:
                return f"{orig_key.split('_')[0].upper()}: {o:.2f} -> {c:.2f}"
            elif o is not None:
                return f"{orig_key.split('_')[0].upper()}: {o:.2f}"
            else:
                return f"{orig_key.split('_')[0].upper()}: -> {c:.2f}"

        if w or wc:
            hud_lines.append("WEATHER:")
            hud_lines.append(f"  {fmt_weather('fog_severity', 'fog')}")
            hud_lines.append(f"  {fmt_weather('rain_severity', 'rain')}")
            hud_lines.append(f"  {fmt_weather('snow_severity', 'snow')}")
        else:
            hud_lines.append("WEATHER: No Data")

        # -------- Helper formatter --------
        def fmt_with_arrow(orig, corr, show_conf=True):
            if not orig:
                return "N/A"

            o_label = orig.get("label", "N/A")
            o_conf = orig.get("confidence")

            c_label = corr.get("label") if corr else None
            c_conf = corr.get("confidence") if corr else None

            if show_conf and o_conf is not None:
                base = f"{o_label} ({o_conf:.2f})"
            else:
                base = f"{o_label}"

            if not corr:
                return base

            if show_conf and c_conf is not None:
                corrected = f"{c_label} ({c_conf:.2f})"
            else:
                corrected = f"{c_label}"

            return f"{base} -> {corrected}"

        # -------- ODD Model --------
        pred = unified.get("prediction") or {}

        if pred:
            hud_lines.append("")
            hud_lines.append("=== ODD MODEL ===")

            hud_lines.append(
                f"TIME: {fmt_with_arrow(pred.get('time'), stage2.get('time_corrected'), False)}"
            )
            hud_lines.append(
                f"SCENE: {fmt_with_arrow(pred.get('scene'), stage2.get('scene_corrected'), False)}"
            )
            hud_lines.append(
                f"VIS: {fmt_with_arrow(pred.get('visibility'), stage2.get('visibility_corrected'))}"
            )
            hud_lines.append(
                f"ROAD: {fmt_with_arrow(pred.get('road_condition_infer'), stage2.get('road_condition_corrected'))}"
            )

            if pred.get("anomalies"):
                hud_lines.append(
                    f"ANOMALY: {pred.get('anomalies').get('label')} ({pred.get('anomalies').get('confidence')})"
                )
        else:
            hud_lines.append("ODD MODEL: No Data")

        # -------- YOLO --------
        if yolo_data:
            hud_lines.append("")
            hud_lines.append(
                f"TRAFFIC: Car {yolo_data.get('car_density',0):.3f} "
                f"Ped {yolo_data.get('pedestrian_density',0):.3f} "
                f"Bike {yolo_data.get('bicycle_density',0):.3f}"
            )
            hud_lines.append(f"WORK_ZONE: {yolo_data.get('work_zone', False)}")

        # -------- Clouds --------
        if "cloud_fraction" in cloud_data:
            hud_lines.append(f"CLOUDS: {cloud_data['cloud_fraction']:.2f}")

        # -------- Glare --------
        if "glare_ratio" in glare_data:
            hud_lines.append(f"GLARE: {glare_data['glare_ratio']:.2f}")

        # -------- Draw --------
        display_img = draw_side_panel(img, hud_lines)

        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, display_img)

    print(f"Done. Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()