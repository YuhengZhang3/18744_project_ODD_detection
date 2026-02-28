import os
import sys
from collections import defaultdict

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
from torch.utils.data import DataLoader

from data.bdd_dataset import (
    BDDDTimeScene,
    collate_time_scene,
    TIME_CLASSES,
    SCENE_CLASSES,
)
from models.odd_model import ODDModel


def eval_time_scene(img_root, label_dir, ckpt_path, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = BDDDTimeScene(img_root=img_root, label_dir=label_dir)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_time_scene,
    )
    print("val size:", len(ds), "batches:", len(loader))

    model = ODDModel(freeze_backbone=True).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    n = 0
    correct_time = 0
    correct_scene = 0

    time_correct = defaultdict(int)
    time_total = defaultdict(int)
    scene_correct = defaultdict(int)
    scene_total = defaultdict(int)

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)

            gt_time = batch["labels"]["time"].to(device)
            gt_scene = batch["labels"]["scene"].to(device)

            out = model(imgs)

            pred_time = out["time"].argmax(dim=1)
            pred_scene = out["scene"].argmax(dim=1)

            correct_time += (pred_time == gt_time).sum().item()
            correct_scene += (pred_scene == gt_scene).sum().item()
            n += imgs.size(0)

            for g, p in zip(gt_time.tolist(), pred_time.tolist()):
                time_total[g] += 1
                if g == p:
                    time_correct[g] += 1

            for g, p in zip(gt_scene.tolist(), pred_scene.tolist()):
                scene_total[g] += 1
                if g == p:
                    scene_correct[g] += 1

    overall_time = correct_time / max(n, 1)
    overall_scene = correct_scene / max(n, 1)

    print("overall time acc:", overall_time)
    print("overall scene acc:", overall_scene)

    print("\nper-class time acc:")
    for i, name in enumerate(TIME_CLASSES):
        tot = time_total[i]
        cor = time_correct[i]
        acc = cor / tot if tot > 0 else 0.0
        print(f"  {i} ({name:10s})  acc={acc:.3f}  {cor}/{tot}")

    print("\nper-class scene acc:")
    for i, name in enumerate(SCENE_CLASSES):
        tot = scene_total[i]
        cor = scene_correct[i]
        acc = cor / tot if tot > 0 else 0.0
        print(f"  {i} ({name:15s})  acc={acc:.3f}  {cor}/{tot}")


if __name__ == "__main__":
    img_root = "/home/yuhengz3@andrew.cmu.edu/bdd100k/100k_datasets/100k/val"
    label_dir = "/home/yuhengz3@andrew.cmu.edu/bdd100k/100k_label/100k/val"
    ckpt_path = "checkpoints_time_scene/best.pt"

    eval_time_scene(
        img_root=img_root,
        label_dir=label_dir,
        ckpt_path=ckpt_path,
        batch_size=128,
    )
