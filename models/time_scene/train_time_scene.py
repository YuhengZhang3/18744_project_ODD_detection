import os
import sys

# Add the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

import torch
# Add Subset to the imports
from torch.utils.data import random_split, Subset

from data.bdd_dataset import BDDDTimeScene, collate_time_scene
from models.odd_model import ODDModel
from losses.odd_losses import odd_loss
from utils.common import seed_everything, get_device, ensure_dir
from utils.single_head_utils import (
    make_loader,
    freeze_vit_only,
    move_batch_to_device,
    save_last_and_best,
)
from utils.multitask_data import fill_missing_heads


# ---------------- CONFIGURATION CONSTANTS ----------------
SEED = 42
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 0.05
NUM_WORKERS = 4
# Updated to use the absolute path from project root to be safe
BDD_ROOT = os.path.join(project_root, "datasets", "bdd_subset")
SAVE_DIR = "checkpoints_time_scene"
# ---------------------------------------------------------


def eval_time_scene(model, loader, device):
    model.eval()

    total_time = 0
    correct_time = 0
    total_scene = 0
    correct_scene = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)

            # --- OPTIMIZATION: FAST EVALUATION ---
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = model(batch["images"])

            gt_time = batch["labels"]["time"]
            gt_scene = batch["labels"]["scene"]

            pred_time = out["time"].argmax(dim=1)
            pred_scene = out["scene"].argmax(dim=1)

            total_time += gt_time.numel()
            correct_time += (pred_time == gt_time).sum().item()

            total_scene += gt_scene.numel()
            correct_scene += (pred_scene == gt_scene).sum().item()

    return (
        correct_time / max(total_time, 1),
        correct_scene / max(total_scene, 1),
    )


def main():
    seed_everything(SEED)
    device = get_device()
    ensure_dir(SAVE_DIR)

    # 1. Load the entire flat dataset
    full_dataset = BDDDTimeScene(
        img_root=os.path.join(BDD_ROOT, "images"),
        label_dir=os.path.join(BDD_ROOT, "labels"),
    )

    # 2. Split into Train (80%) and Val (20%)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Use a generator with the seed to ensure the split is identical across runs
    train_set, val_set = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = make_loader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_time_scene,
    )
    val_loader = make_loader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_time_scene,
    )

    model = ODDModel(freeze_backbone=False).to(device)

    # keep old behavior:
    # freeze vit, but adapter remains trainable
    freeze_vit_only(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best_score = -1.0

    print("device:", device)
    print("total dataset size:", total_size)
    print("train size:", len(train_set))
    print("val size:", len(val_set))

    # --- OPTIMIZATION: MIXED PRECISION SCALER ---
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            batch = fill_missing_heads(batch)

            optimizer.zero_grad()

            # --- OPTIMIZATION: MIXED PRECISION FORWARD PASS ---
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = model(batch["images"])
                loss = odd_loss(out, batch)

            # --- OPTIMIZATION: SCALED BACKWARD PASS ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach())

        scheduler.step()

        train_loss = running_loss / max(len(train_loader), 1)
        val_time_acc, val_scene_acc = eval_time_scene(model, val_loader, device)
        score = 0.5 * val_time_acc + 0.5 * val_scene_acc

        print(
            f"[epoch {epoch}/{EPOCHS}] "
            f"train_loss {train_loss:.4f} "
            f"time_acc {val_time_acc:.4f} "
            f"scene_acc {val_scene_acc:.4f} "
            f"score {score:.4f}"
        )

        config_args = {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "num_workers": NUM_WORKERS,
            "bdd_root": BDD_ROOT,
            "save_dir": SAVE_DIR,
        }

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "time_acc": val_time_acc,
            "scene_acc": val_scene_acc,
            "score": score,
            "args": config_args,
        }

        best_score, is_best = save_last_and_best(
            state=state,
            save_dir=SAVE_DIR,
            score=score,
            best_score=best_score,
        )
        if is_best:
            print("saved best to", os.path.join(SAVE_DIR, "best.pt"))


if __name__ == "__main__":
    main()