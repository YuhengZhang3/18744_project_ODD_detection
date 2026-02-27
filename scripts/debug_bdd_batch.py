import os
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
from torch.utils.data import DataLoader

from data.bdd_dataset import BDDDTimeScene, collate_time_scene
from models.odd_model import ODDModel
from losses.odd_losses import odd_loss


def main():
    img_root = "/Users/zhangyuheng/Documents/Study/CMU_Courses/2026Spring/18744/Project/datasets/100k_datasets/train"
    label_dir = "/Users/zhangyuheng/Documents/Study/CMU_Courses/2026Spring/18744/Project/datasets/100k_label/train"

    ds = BDDDTimeScene(img_root=img_root, label_dir=label_dir)
    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_time_scene,
    )

    batch = next(iter(loader))

    print("images:", batch["images"].shape)
    for k, v in batch["labels"].items():
        print("label", k, v.shape, v[:4])

    model = ODDModel(freeze_backbone=True)
    model.eval()

    with torch.no_grad():
        out = model(batch["images"])
        print("model heads:", list(out.keys()))
        loss = odd_loss(out, batch)
        print("loss:", float(loss))


if __name__ == "__main__":
    main()
