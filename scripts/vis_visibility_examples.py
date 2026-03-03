import os
import json
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_label_map(label_dir, max_per_group=8):
    # collect samples for a few simple groups
    groups = {
        "clear_day": [],
        "clear_night": [],
        "rainy": [],
        "foggy": [],
        "snowy": [],
    }

    files = [f for f in os.listdir(label_dir) if f.endswith(".json")]
    files.sort()
    random.shuffle(files)

    for fname in files:
        path = os.path.join(label_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)

        attrs = data.get("attributes", {})
        weather = attrs.get("weather", "undefined")
        tod = attrs.get("timeofday", "undefined")
        img_name = data.get("name", None)
        if img_name is None:
            continue

        # bdd100k name usually has no suffix, add .jpg
        if not (img_name.endswith(".jpg") or img_name.endswith(".png")):
            img_name = img_name + ".jpg"

        g = None
        if weather == "clear" and tod == "daytime":
            g = "clear_day"
        elif weather == "clear" and tod == "night":
            g = "clear_night"
        elif weather == "rainy":
            g = "rainy"
        elif weather == "foggy":
            g = "foggy"
        elif weather == "snowy":
            g = "snowy"

        if g is None:
            continue

        if len(groups[g]) < max_per_group:
            groups[g].append(
                {
                    "img_name": img_name,
                    "weather": weather,
                    "timeofday": tod,
                }
            )

        if all(len(v) >= max_per_group for v in groups.values()):
            break

    return groups


def calc_edge_contrast(img_bgr):
    # gray for cv features
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img_gray, 100, 200)
    edge_density = float((edges > 0).sum()) / float(edges.size + 1e-6)

    contrast = float(img_gray.std()) / 255.0

    return edge_density, contrast


def show_grid(img_root, groups, out_path="visibility_examples.png", max_per_group=6):
    group_names = [g for g in groups.keys() if len(groups[g]) > 0]
    n_groups = len(group_names)
    if n_groups == 0:
        print("no samples found")
        return

    n_cols = max_per_group
    n_rows = n_groups

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    printed_example = False

    for r, g in enumerate(group_names):
        samples = groups[g][:max_per_group]
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(samples):
                ax.axis("off")
                continue

            info = samples[c]
            img_path = os.path.join(img_root, info["img_name"])
            if not os.path.exists(img_path):
                ax.axis("off")
                continue

            if not printed_example:
                print("example img path:", img_path)
                printed_example = True

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                ax.axis("off")
                continue

            edge_d, contrast = calc_edge_contrast(img_bgr)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            ax.imshow(img_rgb)
            title = (
                f"{g}\n"
                f"{info['weather']}, {info['timeofday']}\n"
                f"edge={edge_d:.3f}, ctr={contrast:.3f}"
            )
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print("saved to", out_path)


def main():
    data_root = "/Users/zhangyuheng/Documents/Study/CMU_Courses/2026Spring/18744/Project/datasets"
    img_root = os.path.join(data_root, "100k_datasets", "val")
    label_root = os.path.join(data_root, "100k_label", "val")

    random.seed(0)

    groups = load_label_map(label_root, max_per_group=8)
    for g, lst in groups.items():
        print(g, len(lst))

    show_grid(img_root, groups, out_path="visibility_examples_val.png", max_per_group=6)


if __name__ == "__main__":
    main()
