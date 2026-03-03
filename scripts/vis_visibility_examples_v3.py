import os
import json
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_label_map(label_dir, max_per_group=8):
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


def calc_features(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    edges_g = cv2.Canny(img_gray, 100, 200)
    edge_g = float((edges_g > 0).sum()) / float(edges_g.size + 1e-6)
    ctr_g = float(img_gray.std()) / 255.0

    x0 = int(w * 0.25)
    x1 = int(w * 0.75)

    y0_near = int(h * 0.5)
    y1_near = int(h * 0.9)
    roi_near = img_gray[y0_near:y1_near, x0:x1]
    edges_near = cv2.Canny(roi_near, 100, 200)
    edge_near = float((edges_near > 0).sum()) / float(edges_near.size + 1e-6)
    ctr_near = float(roi_near.std()) / 255.0

    y0_far = int(h * 0.2)
    y1_far = int(h * 0.5)
    roi_far = img_gray[y0_far:y1_far, x0:x1]
    edges_far = cv2.Canny(roi_far, 100, 200)
    edge_far = float((edges_far > 0).sum()) / float(edges_far.size + 1e-6)
    ctr_far = float(roi_far.std()) / 255.0

    sky = img_gray[0:int(h * 0.3), :]
    ctr_sky = float(sky.std()) / 255.0

    dyn = (float(img_gray.max()) - float(img_gray.min())) / 255.0

    return edge_g, ctr_g, edge_near, ctr_near, edge_far, ctr_far, ctr_sky, dyn


def show_grid(img_root, groups, out_path="visibility_examples_v3_val.png", max_per_group=6):
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
    stats = {g: [] for g in group_names}

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

            edge_g, ctr_g, edge_near, ctr_near, edge_far, ctr_far, ctr_sky, dyn = calc_features(img_bgr)
            stats[g].append(
                [edge_g, ctr_g, edge_near, ctr_near, edge_far, ctr_far, ctr_sky, dyn]
            )

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            title = (
                f"{g}\n"
                f"{info['weather']}, {info['timeofday']}\n"
                f"Gc={ctr_g:.3f}, Ln={ctr_near:.3f}\n"
                f"F={ctr_far:.3f}, Sky={ctr_sky:.3f}"
            )
            ax.imshow(img_rgb)
            ax.set_title(title, fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print("saved to", out_path)

    print("group mean features:")
    for g in group_names:
        arr = np.array(stats[g])
        edge_g_m, ctr_g_m, edge_near_m, ctr_near_m, edge_far_m, ctr_far_m, ctr_sky_m, dyn_m = arr.mean(axis=0)
        print(
            f"{g:11s}  "
            f"G c={ctr_g_m:.3f}  "
            f"L near c={ctr_near_m:.3f}  "
            f"F c={ctr_far_m:.3f}  "
            f"Sky c={ctr_sky_m:.3f}  "
            f"dyn={dyn_m:.3f}"
        )


def main():
    data_root = "/Users/zhangyuheng/Documents/Study/CMU_Courses/2026Spring/18744/Project/datasets"
    img_root = os.path.join(data_root, "100k_datasets", "val")
    label_root = os.path.join(data_root, "100k_label", "val")

    random.seed(0)

    groups = load_label_map(label_root, max_per_group=8)
    for g, lst in groups.items():
        print(g, len(lst))

    show_grid(img_root, groups, out_path="visibility_examples_v3_val.png", max_per_group=6)


if __name__ == "__main__":
    main()
