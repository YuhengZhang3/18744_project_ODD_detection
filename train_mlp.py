"""
Train the Stage-2 Tiebreaker MLP.

Usage:
    python train_tiebreaker.py
    python train_tiebreaker.py --data data/tiebreaker_train.pt --epochs 100 --lr 1e-3
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from models.tiebreaker.tiebreaker_mlp import TiebreakerMLP


# ============================================================
# Dataset
# ============================================================

class TiebreakerDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ============================================================
# Masked Loss
# ============================================================

def compute_loss(outputs, Y):
    """
    Compute masked loss across all heads.
    Y columns: [fog, rain, snow, time, scene, anomalies]
    Values of -1 are masked out.
    """
    losses = {}
    total = torch.tensor(0.0, device=Y.device)

    # --- Weather heads: BCE with masking ---
    for i, name in enumerate(["fog", "rain", "snow"]):
        target = Y[:, i]
        mask = target != -1

        if mask.sum() > 0:
            pred = outputs[name][mask]
            tgt = target[mask]
            loss = F.binary_cross_entropy_with_logits(pred, tgt)
            losses[name] = loss
            total = total + loss

    # --- Classification heads: CE with ignore_index ---
    ce_heads = [
        ("time", 3, 4),       # Y column 3, num_classes 4
        ("scene", 4, 7),      # Y column 4, num_classes 7
        ("anomalies", 5, 4),  # Y column 5, num_classes 4
    ]

    for name, col_idx, num_classes in ce_heads:
        target = Y[:, col_idx].long()

        # Replace -1 with a proper ignore index (CE uses -100 by default)
        target_ce = target.clone()
        target_ce[target == -1] = -100

        if (target_ce != -100).sum() > 0:
            pred = outputs[name]
            loss = F.cross_entropy(pred, target_ce, ignore_index=-100)
            losses[name] = loss
            total = total + loss

    return total, losses


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    head_correct = {"fog": 0, "rain": 0, "snow": 0, "time": 0, "scene": 0, "anomalies": 0}
    head_total = {"fog": 0, "rain": 0, "snow": 0, "time": 0, "scene": 0, "anomalies": 0}
    n_batches = 0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        outputs = model(X_batch)
        loss, _ = compute_loss(outputs, Y_batch)
        total_loss += loss.item()
        n_batches += 1

        # Accuracy for weather heads
        for i, name in enumerate(["fog", "rain", "snow"]):
            target = Y_batch[:, i]
            mask = target != -1
            if mask.sum() > 0:
                pred = (torch.sigmoid(outputs[name][mask]) > 0.5).float()
                head_correct[name] += (pred == target[mask]).sum().item()
                head_total[name] += mask.sum().item()

        # Accuracy for classification heads
        for name, col_idx in [("time", 3), ("scene", 4), ("anomalies", 5)]:
            target = Y_batch[:, col_idx].long()
            mask = target != -1
            if mask.sum() > 0:
                pred = outputs[name][mask].argmax(dim=1)
                head_correct[name] += (pred == target[mask]).sum().item()
                head_total[name] += mask.sum().item()

    avg_loss = total_loss / max(n_batches, 1)

    accuracies = {}
    for name in head_correct:
        if head_total[name] > 0:
            accuracies[name] = head_correct[name] / head_total[name]
        else:
            accuracies[name] = None  # no valid samples

    return avg_loss, accuracies


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data = torch.load(args.data, map_location="cpu")
    X, Y = data["X"], data["Y"]
    print(f"Loaded: X={X.shape}, Y={Y.shape}")

    # Train/val split
    dataset = TiebreakerDataset(X, Y)
    n_val = int(len(dataset) * args.val_ratio)
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train: {n_train}, Val: {n_val}")

    # Model
    model = TiebreakerMLP(
        input_dim=X.shape[1],
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # Training loop
    best_val_loss = float("inf")
    ckpt_dir = os.path.join(project_root, args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_head_losses = {}
        n_batches = 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            outputs = model(X_batch)
            loss, head_losses = compute_loss(outputs, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            for k, v in head_losses.items():
                epoch_head_losses[k] = epoch_head_losses.get(k, 0.0) + v.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        # Print
        if epoch % args.print_every == 0 or epoch == 1:
            acc_str = " | ".join(
                f"{k}={v:.3f}" if v is not None else f"{k}=N/A"
                for k, v in val_acc.items()
            )
            head_str = " | ".join(
                f"{k}={v / max(n_batches, 1):.4f}"
                for k, v in epoch_head_losses.items()
            )
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc: {acc_str}"
            )
            if epoch % (args.print_every * 5) == 0:
                print(f"  head losses: {head_str}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(ckpt_dir, "tiebreaker_best.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)
            print(f"  >> Saved best model (val_loss={val_loss:.4f})")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final evaluation on val set:")
    val_loss, val_acc = evaluate(model, val_loader, device)
    for name, acc in val_acc.items():
        if acc is not None:
            print(f"  {name:12s}: {acc:.4f}")
        else:
            print(f"  {name:12s}: N/A (no valid samples)")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {ckpt_dir}/tiebreaker_best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train Tiebreaker MLP")

    # Data
    parser.add_argument("--data", type=str, default="data/tiebreaker_train.pt")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--hidden1", type=int, default=128)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Output
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_tiebreaker")
    parser.add_argument("--print_every", type=int, default=5)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()