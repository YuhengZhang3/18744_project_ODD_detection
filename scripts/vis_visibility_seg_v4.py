import os
import json
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


# ---------------------------------------
# segformer loading (road segmentation)
# ---------------------------------------
SEG_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_image_processor = AutoImageProcessor.from_pretrained(SEG_MODEL_NAME)
_seg_model = AutoModelForSemanticSegmentation.from_pretrained(SEG_MODEL_NAME).to(_device)
_seg_model.eval()


def get_segformer_road_features(img_bgr):
    """
    run segformer, get road mask stats:
    - road contrast (gray std over road pixels)
    - road edge density (canny edges inside road)
    - road depth (how far road extends vertically, normalized 0~1)
    """
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    inputs = _image_processor(images=img_rgb, return_tensors="pt").to(_device)

    with torch.no_grad():
        outputs = _seg_model(**inputs)
        logits = outputs.logits  # (1, C, h', w')

    # upsample to original size
    logits = F.interpolate(
        logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )
    preds = logits.argmax(dim=1)[0].cpu().numpy()  # (h, w)

    # simple road mask: cityscapes road + sidewalk (0, 1)
    road_mask = (preds == 0) | (preds == 1)

    if road_mask.sum() == 0:
        return 0.0, 0.0, 0.0

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    road_pixels = img_gray[road_mask]
    road_contrast = float(road_pixels.std()) / 255.0

    edges = cv2.Canny(img_gray, 100, 200)
    road_edges = edges[road_mask]
    road_edge_density = float((road_edges > 0).sum()) / float(road_edges.size + 1e-6)

    rows = np.where(road_mask.any(axis=1))[0]
    if len(rows) == 0:
        road_depth = 0.0
    else:
        top_row = rows.min()
        bottom_row = rows.max()
        visible_height = bottom_row - top_row + 1
        road_depth = float(visible_height) / float(h)

    return road_contrast, road_edge_density, road_depth


# ---------------------------------------
# bdd grouping
# ---------------------------------------

def load_label_map(label_dir, max_per_group=4):
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


# ---------------------------------------
# CV features
# ---------------------------------------

def calc_features(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    # global contrast
    edges_g = cv2.Canny(img_gray, 100, 200)
    edge_g = float((edges_g > 0).sum()) / float(edges_g.size + 1e-6)
    ctr_g = float(img_gray.std()) / 255.0

    # near-road roi
    x0 = int(w * 0.25)
    x1 = int(w * 0.75)
    y0_near = int(h * 0.5)
    y1_near = int(h * 0.9)
    roi_near = img_gray[y0_near:y1_near, x0:x1]
    edges_near = cv2.Canny(roi_near, 100, 200)
    edge_near = float((edges_near > 0).sum()) / float(edges_near.size + 1e-6)
    ctr_near = float(roi_near.std()) / 255.0

    # far-road roi
    y0_far = int(h * 0.2)
    y1_far = int(h * 0.5)
    roi_far = img_gray[y0_far:y1_far, x0:x1]
    edges_far = cv2.Canny(roi_far, 100, 200)
    edge_far = float((edges_far > 0).sum()) / float(edges_far.size + 1e-6)
    ctr_far = float(roi_far.std()) / 255.0

    # sky / top
    sky = img_gray[0:int(h * 0.3), :]
    ctr_sky = float(sky.std()) / 255.0

    # laplacian blur (global + near + far)
    lap_g = cv2.Laplacian(img_gray, cv2.CV_64F)
    blur_g = float(lap_g.var()) / (255.0 ** 2 + 1e-6)

    lap_near = cv2.Laplacian(roi_near, cv2.CV_64F)
    blur_near = float(lap_near.var()) / (255.0 ** 2 + 1e-6)

    lap_far = cv2.Laplacian(roi_far, cv2.CV_64F)
    blur_far = float(lap_far.var()) / (255.0 ** 2 + 1e-6)

    # simple dark-channel haze score (whole image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    dark = np.min(img_rgb, axis=2)
    haze_score = float(dark.mean()) / 255.0

    dyn = (float(img_gray.max()) - float(img_gray.min())) / 255.0

    # segformer road features
    road_contrast, road_edge_density, road_depth = get_segformer_road_features(img_bgr)

    return [
        edge_g,
        ctr_g,
        edge_near,
        ctr_near,
        edge_far,
        ctr_far,
        ctr_sky,
        blur_g,
        blur_near,
        blur_far,
        haze_score,
        dyn,
        road_contrast,
        road_edge_density,
        road_depth,
    ]


def show_grid(img_root, groups, out_path="visibility_examples_v4_val.png", max_per_group=4):
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

            feats = calc_features(img_bgr)
            stats[g].append(feats)

            (
                edge_g,
                ctr_g,
                edge_near,
                ctr_near,
                edge_far,
                ctr_far,
                ctr_sky,
                blur_g,
                blur_near,
                blur_far,
                haze_score,
                dyn,
                road_contrast,
                road_edge_density,
                road_depth,
            ) = feats

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            title = (
                f"{g}\n"
                f"{info['weather']}, {info['timeofday']}\n"
                f"Gc={ctr_g:.3f}, Ln={ctr_near:.3f}, F={ctr_far:.3f}\n"
                f"Sky={ctr_sky:.3f}, RoadCtr={road_contrast:.3f}, depth={road_depth:.2f}"
            )
            ax.imshow(img_rgb)
            ax.set_title(title, fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print("saved to", out_path)

    # print group mean stats
    print("group mean features:")
    for g in group_names:
        arr = np.array(stats[g])
        (
            edge_g_m,
            ctr_g_m,
            edge_near_m,
            ctr_near_m,
            edge_far_m,
            ctr_far_m,
            ctr_sky_m,
            blur_g_m,
            blur_near_m,
            blur_far_m,
            haze_m,
            dyn_m,
            road_ctr_m,
            road_edge_m,
            road_depth_m,
        ) = arr.mean(axis=0)

        print(
            f"{g:11s}  "
            f"G c={ctr_g_m:.3f}  "
            f"L near c={ctr_near_m:.3f}  "
            f"F c={ctr_far_m:.3f}  "
            f"Sky c={ctr_sky_m:.3f}  "
            f"blurN={blur_near_m:.4f}  "
            f"haze={haze_m:.3f}  "
            f"roadCtr={road_ctr_m:.3f}  "
            f"roadDepth={road_depth_m:.3f}"
        )


def main():
    data_root = "/Users/zhangyuheng/Documents/Study/CMU_Courses/2026Spring/18744/Project/datasets"
    img_root = os.path.join(data_root, "100k_datasets", "val")
    label_root = os.path.join(data_root, "100k_label", "val")

    random.seed(0)

    groups = load_label_map(label_root, max_per_group=4)
    for g, lst in groups.items():
        print(g, len(lst))

    show_grid(img_root, groups, out_path="visibility_examples_v4_val.png", max_per_group=4)


if __name__ == "__main__":
    main()
