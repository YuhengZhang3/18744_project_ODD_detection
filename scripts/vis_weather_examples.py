import os
import cv2
import json
import glob

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

IMAGE_DIR = os.path.join(project_root, "source_images")
MERGED_DIR = os.path.join(project_root, "outputs", "merged_json")

TARGETS = [0.0, 0.3, 0.6, 0.9]
TOLERANCE = 0.15  # adjustable


def load_weather(basename):
    json_path = os.path.join(MERGED_DIR, f"{basename}.json")
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            weather = data.get("weather", {})
            return weather.get("weather", weather)
    except:
        return None


def collect_images():
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext.upper())))
    return sorted(image_paths)


def find_best_matches(image_paths):
    categories = ["fog_severity", "rain_severity", "snow_severity"]

    best = {
        cat: {t: (None, float("inf")) for t in TARGETS}
        for cat in categories
    }

    for path in image_paths:
        basename = os.path.splitext(os.path.basename(path))[0]
        weather = load_weather(basename)

        if not weather:
            continue

        for cat in categories:
            if cat not in weather:
                continue

            val = weather[cat]

            for t in TARGETS:
                diff = abs(val - t)
                if diff < TOLERANCE and diff < best[cat][t][1]:
                    best[cat][t] = (path, diff)

    # Flatten results
    selected = []
    for cat in categories:
        for t in TARGETS:
            path, diff = best[cat][t]
            if path is not None:
                selected.append((cat, t, path))

    return selected


def draw_weather_hud(img, category, target, actual):
    text = f"{category.replace('_severity','').upper()}: {actual:.2f}"

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    cv2.putText(
        img,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return img


def main():
    image_paths = collect_images()

    if not image_paths:
        print("No images found.")
        return

    selected = find_best_matches(image_paths)

    if not selected:
        print("No matching weather samples found.")
        return

    print(f"Loaded {len(selected)} samples")

    idx = 0

    while True:
        category, target, path = selected[idx]

        img = cv2.imread(path)
        if img is None:
            idx = (idx + 1) % len(selected)
            continue

        basename = os.path.splitext(os.path.basename(path))[0]
        weather = load_weather(basename)

        actual = weather.get(category, 0.0) if weather else 0.0

        # Resize
        max_h = 800
        if img.shape[0] > max_h:
            scale = max_h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

        img = draw_weather_hud(img, category, target, actual)

        cv2.imshow("Weather Subset Viewer", img)

        key = cv2.waitKey(0) & 0xFF

        if key in [ord('q'), 27]:
            break
        elif key in [ord('d'), 83]:
            idx = (idx + 1) % len(selected)
        elif key in [ord('a'), 81]:
            idx = (idx - 1) % len(selected)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()