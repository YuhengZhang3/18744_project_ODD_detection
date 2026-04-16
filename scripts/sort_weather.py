import os
import json
import glob
import shutil

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

IMAGE_DIR = os.path.join(project_root, "source_images")
MERGED_DIR = os.path.join(project_root, "outputs", "merged_json")
OUTPUT_DIR = os.path.join(project_root, "output", "weather_examples")

THRESHOLD = 0.05  # ignore near-zero noise


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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def severity_bucket(val):
    """
    Convert continuous severity into ordered buckets
    """
    if val < 0.2:
        return "0_low"
    elif val < 0.5:
        return "1_mid"
    elif val < 0.8:
        return "2_high"
    else:
        return "3_extreme"


def save_images(image_paths):
    counts = {"fog": 0, "rain": 0, "snow": 0}

    for path in image_paths:
        filename = os.path.basename(path)
        basename = os.path.splitext(filename)[0]

        weather = load_weather(basename)
        if not weather:
            continue

        categories = {
            "fog": weather.get("fog_severity", 0.0),
            "rain": weather.get("rain_severity", 0.0),
            "snow": weather.get("snow_severity", 0.0),
        }

        for label, val in categories.items():
            if val < THRESHOLD:
                continue

            bucket = severity_bucket(val)

            out_dir = os.path.join(OUTPUT_DIR, label, bucket)
            ensure_dir(out_dir)

            # include severity in filename for sorting/debugging
            new_name = f"{val:.2f}_{filename}"
            dst_path = os.path.join(out_dir, new_name)

            shutil.copy2(path, dst_path)
            counts[label] += 1

    print("\nSaved image counts:")
    for k, v in counts.items():
        print(f"{k}: {v}")


def main():
    image_paths = collect_images()

    if not image_paths:
        print("No images found.")
        return

    print("Sorting images into weather categories...")
    save_images(image_paths)

    print("\nDone.")


if __name__ == "__main__":
    main()