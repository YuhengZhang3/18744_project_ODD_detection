import os
import sys
from time import time

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
from torch.utils.data import DataLoader

from data.bdd_dataset import BDDDTimeScene, collate_time_scene
from models.odd_model import ODDModel
from losses.odd_losses import odd_loss


def train():
    img_root = "/Users/zhangyuheng/Documents/Study/CMU_Courses/2026Spring/18744/Project/datasets/100k_datasets/train"
    label_dir = "/Users/zhangyuheng/Documents/Study/CMU_Courses/2026Spring/18744/Project/datasets/100k_label/train"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = BDDDTimeScene(img_root=img_root, label_dir=label_dir)
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_time_scene,
    )

    model = ODDModel(freeze_backbone=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    max_steps = 20
    log_every = 5

    step = 0
    t0 = time()
    while step < max_steps:
        for batch in loader:
            step += 1
            if step > max_steps:
                break

            imgs = batch["images"].to(device)

            for k in batch["labels"]:
                batch["labels"][k] = batch["labels"][k].to(device)
                batch["mask"][k] = batch["mask"][k].to(device)
                batch["severity"][k] = batch["severity"][k].to(device)

            model.train()
            out = model(imgs)
            loss = odd_loss(out, batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % log_every == 0:
                with torch.no_grad():
                    preds_time = out["time"].argmax(dim=1)
                    preds_scene = out["scene"].argmax(dim=1)
                    acc_time = (preds_time == batch["labels"]["time"]).float().mean()
                    acc_scene = (preds_scene == batch["labels"]["scene"]).float().mean()

                dt = time() - t0
                print(
                    f"step {step}  "
                    f"loss {float(loss):.4f}  "
                    f"time_acc {float(acc_time):.3f}  "
                    f"scene_acc {float(acc_scene):.3f}  "
                    f"({dt:.1f}s)"
                )
                t0 = time()

    torch.save(model.state_dict(), "odd_time_scene_bdd.pt")
    print("saved to odd_time_scene_bdd.pt")


if __name__ == "__main__":
    train()
