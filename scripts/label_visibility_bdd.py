import os
import json
import cv2
import numpy as np
from tqdm import tqdm


def get_data_root():
    candidates = [
        "/home/yuhengz3@andrew.cmu.edu/bdd100k",
        "/Users/zhangyuheng/Documents/Study/CMU_Courses/2026Spring/18744/Project/datasets",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise RuntimeError(
        "data_root not found. Please update candidates in get_data_root()."
    )


def get_near_far_contrast(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    x0 = int(w * 0.25)
    x1 = int(w * 0.75)

    y0_near = int(h * 0.5)
    y1_near = int(h * 0.9)
    near = img_gray[y0_near:y1_near, x0:x1]

    y0_far = int(h * 0.2)
    y1_far = int(h * 0.5)
    far = img_gray[y0_far:y1_far, x0:x1]

    near_c = float(near.std()) / 255.0
    far_c = float(far.std()) / 255.0

    return near_c, far_c


def assign_visibility(weather, tod, near_c, far_c):
    # 2 = good, 1 = medium, 0 = poor
    fog_near_thr = 0.135
    fog_far_thr = 0.17
    night_far_thr = 0.19

    if weather == "foggy":
        return 0

    if near_c < fog_near_thr and far_c < fog_far_thr:
        return 0

    if tod == "night" and far_c < night_far_thr and near_c >= fog_near_thr:
        return 1

    return 2


def resolve_split_dir(root, base, split):
    # base is "100k_datasets" or "100k_label"
    # support both:
    #   root/base/split
    #   root/base/100k/split
    cand1 = os.path.join(root, base, split)
    cand2 = os.path.join(root, base, "100k", split)
    for c in (cand1, cand2):
        if os.path.isdir(c):
            return c
    raise RuntimeError(f"no dir found for {base}/{split} under {root}")


def process_split(data_root, split):
    img_dir = resolve_split_dir(data_root, "100k_datasets", split)
    label_dir = resolve_split_dir(data_root, "100k_label", split)

    out_dir = os.path.join(data_root, "visibility_labels", split)
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(label_dir) if f.endswith(".json")]
    files.sort()

    print(f"split {split}, img_dir={img_dir}, label_dir={label_dir}, num labels {len(files)}")

    for fname in tqdm(files):
        jpath = os.path.join(label_dir, fname)
        with open(jpath, "r") as f:
            data = json.load(f)

        attrs = data.get("attributes", {})
        weather = attrs.get("weather", "undefined")
        tod = attrs.get("timeofday", "undefined")

        img_name = data.get("name", None)
        if img_name is None:
            continue

        if not (img_name.endswith(".jpg") or img_name.endswith(".png")):
            img_name = img_name + ".jpg"

        ipath = os.path.join(img_dir, img_name)
        if not os.path.exists(ipath):
            continue

        img = cv2.imread(ipath)
        if img is None:
            continue

        near_c, far_c = get_near_far_contrast(img)
        vis = assign_visibility(weather, tod, near_c, far_c)

        out = {
            "name": img_name,
            "visibility": int(vis),
            "weather": weather,
            "timeofday": tod,
            "near_contrast": near_c,
            "far_contrast": far_c,
        }

        out_name = os.path.splitext(fname)[0] + "_vis.json"
        opath = os.path.join(out_dir, out_name)
        with open(opath, "w") as f:
            json.dump(out, f)


def main():
    data_root = get_data_root()
    print("using data_root:", data_root)
    for split in ["train", "val", "test"]:
        process_split(data_root, split)


if __name__ == "__main__":
    main()