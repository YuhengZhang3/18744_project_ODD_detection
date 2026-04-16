"""
Compare Stage-1 vs Stage-2 Tiebreaker accuracy on the same val split.

Usage:
    python eval_tiebreaker.py
    python eval_tiebreaker.py --data data/tiebreaker_train.pt --ckpt checkpoints_tiebreaker/tiebreaker_best.pt
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from models.tiebreaker.tiebreaker_mlp import TiebreakerMLP


class TiebreakerDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/tiebreaker_train.pt")
    parser.add_argument("--ckpt", type=str, default="checkpoints_tiebreaker/tiebreaker_best.pt")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary weather predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data with same split as training
    data = torch.load(args.data, map_location="cpu")
    X, Y = data["X"], data["Y"]

    dataset = TiebreakerDataset(X, Y)
    n_val = int(len(dataset) * args.val_ratio)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(args.seed)
    _, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    # Extract val data
    val_indices = val_set.indices
    print(f"First 5 val indices: {val_indices[:5]}")
    X_val = X[val_indices]
    Y_val = Y[val_indices]

    print(f"Val set size: {len(val_indices)}")
    print(f"Threshold for binary heads: {args.threshold}")

    # ============================================================
    # Stage-1 predictions (extracted from X)
    # ============================================================
    # X layout:
    #   [0]     cloud_fraction
    #   [1]     fog_severity
    #   [2]     rain_severity
    #   [3]     snow_severity
    #   [4]     glare_ratio
    #   [5:9]   time softmax (4)
    #   [9:16]  scene softmax (7)
    #   [16:19] visibility softmax (3)
    #   [19:23] anomalies softmax (4)
    #   [23:50] road_condition softmax (27)
    #   [50:]   virtual sensors

    s1_fog = (X_val[:, 1] > args.threshold).float()
    s1_rain = (X_val[:, 2] > args.threshold).float()
    s1_snow = (X_val[:, 3] > args.threshold).float()
    s1_time = X_val[:, 5:9].argmax(dim=1)
    s1_scene = X_val[:, 9:16].argmax(dim=1)

    # ============================================================
    # Stage-2 predictions (MLP)
    # ============================================================
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = TiebreakerMLP(input_dim=X.shape[1])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(X_val.to(device))

    s2_fog = (torch.sigmoid(outputs["fog"].cpu()) > args.threshold).float()
    s2_rain = (torch.sigmoid(outputs["rain"].cpu()) > args.threshold).float()
    s2_snow = (torch.sigmoid(outputs["snow"].cpu()) > args.threshold).float()
    s2_time = outputs["time"].cpu().argmax(dim=1)
    s2_scene = outputs["scene"].cpu().argmax(dim=1)

    # ============================================================
    # Compare
    # ============================================================

    print("\n" + "=" * 70)
    print(f"  {'HEAD':<14s} {'STAGE-1':>10s} {'STAGE-2':>10s} {'DELTA':>10s}   VERDICT")
    print("=" * 70)

    comparisons = [
        ("fog",   s1_fog,   s2_fog,   Y_val[:, 0]),
        ("rain",  s1_rain,  s2_rain,  Y_val[:, 1]),
        ("snow",  s1_snow,  s2_snow,  Y_val[:, 2]),
        ("time",  s1_time,  s2_time,  Y_val[:, 3]),
        ("scene", s1_scene, s2_scene, Y_val[:, 4]),
    ]

    for name, s1_pred, s2_pred, target in comparisons:
        if name in ("fog", "rain", "snow"):
            mask = target != -1
        else:
            mask = target.long() != -1

        if mask.sum() == 0:
            print(f"  {name:<14s} {'N/A':>10s} {'N/A':>10s} {'':>10s}   (no valid samples)")
            continue

        tgt = target[mask]
        if name in ("time", "scene"):
            tgt = tgt.long()

        s1_acc = (s1_pred[mask] == tgt).float().mean().item()
        s2_acc = (s2_pred[mask] == tgt).float().mean().item()
        delta = s2_acc - s1_acc

        if delta > 0.001:
            verdict = f"IMPROVED (+{delta*100:.2f}%)"
        elif delta < -0.001:
            verdict = f"REGRESSED ({delta*100:.2f}%)"
        else:
            verdict = "UNCHANGED"

        print(f"  {name:<14s} {s1_acc:>10.4f} {s2_acc:>10.4f} {delta:>+10.4f}   {verdict}")

    # ============================================================
    # Per-class breakdown for weather
    # ============================================================
    print("\n" + "-" * 70)
    print("Weather per-class breakdown (precision / recall):")
    print("-" * 70)

    for i, name in enumerate(["fog", "rain", "snow"]):
        target = Y_val[:, i]
        mask = target != -1
        if mask.sum() == 0:
            continue

        tgt = target[mask]
        s1 = [s1_fog, s1_rain, s1_snow][i][mask]
        s2 = [s2_fog, s2_rain, s2_snow][i][mask]

        n_pos = (tgt == 1).sum().item()
        n_neg = (tgt == 0).sum().item()

        for stage_name, pred in [("Stage-1", s1), ("Stage-2", s2)]:
            tp = ((pred == 1) & (tgt == 1)).sum().item()
            fp = ((pred == 1) & (tgt == 0)).sum().item()
            fn = ((pred == 0) & (tgt == 1)).sum().item()
            tn = ((pred == 0) & (tgt == 0)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            print(f"  {name} {stage_name}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}  (TP={tp} FP={fp} FN={fn} TN={tn})")

    # ============================================================
    # Confusion cases: where S2 fixed S1 vs broke S1
    # ============================================================
    print("\n" + "-" * 70)
    print("Correction analysis (S1 wrong -> S2 fixed vs S1 right -> S2 broke):")
    print("-" * 70)

    for name, s1_pred, s2_pred, target in comparisons:
        if name in ("fog", "rain", "snow"):
            mask = target != -1
        else:
            mask = target.long() != -1

        if mask.sum() == 0:
            continue

        tgt = target[mask]
        if name in ("time", "scene"):
            tgt = tgt.long()

        s1_p = s1_pred[mask]
        s2_p = s2_pred[mask]

        s1_correct = (s1_p == tgt)
        s2_correct = (s2_p == tgt)

        fixed = (~s1_correct & s2_correct).sum().item()   # S1 wrong, S2 right
        broke = (s1_correct & ~s2_correct).sum().item()    # S1 right, S2 wrong
        both_right = (s1_correct & s2_correct).sum().item()
        both_wrong = (~s1_correct & ~s2_correct).sum().item()

        total_valid = mask.sum().item()
        print(
            f"  {name:<10s}: "
            f"FIXED={fixed:5d} | BROKE={broke:5d} | "
            f"both_right={both_right:5d} | both_wrong={both_wrong:5d} | "
            f"net={fixed - broke:+d}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()