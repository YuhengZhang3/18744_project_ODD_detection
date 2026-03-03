import os
import json
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_by_vis(label_dir):
    buckets = {0: [], 1: [], 2: []}
    files = [f for f in os.listdir(label_dir) if f.endswith("_vis.json")]
    for f in files:
        path = os.path.join(label_dir, f)
        with open(path, "r") as fh:
            data = json.load(fh)
        v = int(data.get("visibility", 2))
        if v not in buckets:
            continue
        buckets[v].append(data)
    return buckets


def show_grid(data_root, split="val", per_vis=8, out_path="vis_visibility_samples_val.png"):
    img_dir = os.path.join(data_root, "100k_datasets", split)
    label_dir = os.path.join(data_root, "visibility_labels", split)

    buckets = load_by_vis(label_dir)

    levels = [0, 1, 2]
    names = {0: "poor", 1: "medium", 2: "good"}

    n_rows = len(levels)
    n_cols = per_vis

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    random.seed(0)

    for r, v in enumerate(levels):
        imgs = buckets[v]
        random.shuffle(imgs)
        imgs = imgs[:per_vis]

        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(imgs):
                ax.axis("off")
                continue

            info = imgs[c]
            img_name = info["name"]
            ipath = os.path.join(img_dir, img_name)
            if not os.path.exists(ipath):
                ax.axis("off")
                continue

            img_bgr = cv2.imread(ipath)
            if img_bgr is None:
                ax.axis("off")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            wc = info.get("weather", "na")
            tod = info.get("timeofday", "na")
            nc = info.get("near_contrast", 0.0)
            fc = info.get("far_contrast", 0.0)

            title = (
                f"vis={v} ({names[v]})\n"
                f"{wc}, {tod}\n"
                f"near={nc:.3f}, far={fc:.3f}"
            )

            ax.imshow(img_rgb)
            ax.set_title(title, fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print("saved to", out_path)


def main():
    data_root = "/Users/zhangyuheng/Documents/Study/CMU_Courses/2026Spring/18744/Project/datasets"
    show_grid(data_root, split="val", per_vis=8, out_path="vis_visibility_samples_val.png")


if __name__ == "__main__":
    main()
