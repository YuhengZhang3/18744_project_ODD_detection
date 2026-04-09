import os
import sys
import argparse
from collections import defaultdict

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
from torch.utils.data import DataLoader
from data.rscd_dataset import RSCDRoadCondition, ROAD_STATE_CLASSES, ROAD_SEVERITY_CLASSES
from utils.multitask_data import collate_road_condition
from utils.infer_utils import load_infer_model


def make_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_road_condition,
        drop_last=False,
    )


def update_stats(stats, y_true, y_pred, class_names):
    for t, p in zip(y_true, y_pred):
        t = int(t)
        p = int(p)
        stats[t]["count"] += 1
        stats[t]["correct"] += int(t == p)
        stats[t]["class_name"] = class_names[t]


def summarize(stats, class_names):
    rows = []
    total = 0
    correct = 0
    for i, cname in enumerate(class_names):
        cnt = stats[i]["count"]
        cor = stats[i]["correct"]
        acc = cor / cnt if cnt > 0 else 0.0
        rows.append({
            "class_id": i,
            "class_name": cname,
            "count": cnt,
            "correct": cor,
            "acc": acc,
        })
        total += cnt
        correct += cor
    overall = correct / total if total > 0 else 0.0
    return overall, rows


def format_block(title, overall, rows):
    lines = [title, "overall acc: {:.4f}".format(overall)]
    for r in rows:
        lines.append(
            "  {:>2d} ({:<10s}) acc={:.3f}  {}/{}".format(
                r["class_id"], r["class_name"], r["acc"], r["correct"], r["count"]
            )
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--rscd_root", type=str, default="/home/yuhengz3@andrew.cmu.edu/rscd/dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    model, device, _ = load_infer_model(ckpt_path=args.ckpt_path)

    test_set = RSCDRoadCondition(root=args.rscd_root, split="test")
    loader = make_loader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)

    state_stats = defaultdict(lambda: {"count": 0, "correct": 0, "class_name": ""})
    severity_stats = defaultdict(lambda: {"count": 0, "correct": 0, "class_name": ""})

    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device, non_blocking=True)
            y_state = batch["labels"]["road_state"].to(device, non_blocking=True)
            y_severity = batch["labels"]["road_severity"].to(device, non_blocking=True)

            out = model(imgs)
            pred_state = out["road_state"].argmax(dim=1)
            pred_severity = out["road_severity"].argmax(dim=1)

            update_stats(
                state_stats,
                y_state.detach().cpu().tolist(),
                pred_state.detach().cpu().tolist(),
                ROAD_STATE_CLASSES,
            )
            update_stats(
                severity_stats,
                y_severity.detach().cpu().tolist(),
                pred_severity.detach().cpu().tolist(),
                ROAD_SEVERITY_CLASSES,
            )

    state_overall, state_rows = summarize(state_stats, ROAD_STATE_CLASSES)
    severity_overall, severity_rows = summarize(severity_stats, ROAD_SEVERITY_CLASSES)

    print(format_block("TEST road_state", state_overall, state_rows))
    print("")
    print(format_block("TEST road_severity", severity_overall, severity_rows))


if __name__ == "__main__":
    main()
