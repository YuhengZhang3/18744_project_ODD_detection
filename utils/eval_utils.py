import json
import os
from collections import Counter

import torch


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def counter_from_list(labels):
    c = Counter()
    for y in labels:
        c[int(y)] += 1
    return c


def overall_accuracy(y_true, y_pred):
    total = len(y_true)
    if total == 0:
        return 0.0
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return correct / total


def per_class_accuracy(y_true, y_pred, class_names):
    stats = []
    ncls = len(class_names)

    for i in range(ncls):
        total_i = sum(1 for y in y_true if y == i)
        correct_i = sum(1 for yt, yp in zip(y_true, y_pred) if yt == i and yp == i)
        acc_i = correct_i / total_i if total_i > 0 else 0.0
        stats.append(
            {
                "class_id": i,
                "class_name": class_names[i],
                "count": total_i,
                "correct": correct_i,
                "acc": acc_i,
            }
        )
    return stats


def segmentation_iou_stats(y_true_list, y_pred_list, class_names):
    ncls = len(class_names)
    stats = []

    for cid in range(ncls):
        inter = 0
        union = 0
        true_pixels = 0
        pred_pixels = 0

        for yt, yp in zip(y_true_list, y_pred_list):
            yt_c = (yt == cid)
            yp_c = (yp == cid)
            inter += int((yt_c & yp_c).sum())
            union += int((yt_c | yp_c).sum())
            true_pixels += int(yt_c.sum())
            pred_pixels += int(yp_c.sum())

        iou = inter / union if union > 0 else 0.0
        stats.append(
            {
                "class_id": cid,
                "class_name": class_names[cid],
                "intersection": inter,
                "union": union,
                "true_pixels": true_pixels,
                "pred_pixels": pred_pixels,
                "iou": iou,
            }
        )

    return stats


def mean_iou(stats, include_ids=None):
    vals = []
    for s in stats:
        if include_ids is not None and s["class_id"] not in include_ids:
            continue
        vals.append(s["iou"])
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def foreground_iou_from_seg_lists(y_true_list, y_pred_list):
    inter = 0
    union = 0
    for yt, yp in zip(y_true_list, y_pred_list):
        yt_fg = yt > 0
        yp_fg = yp > 0
        inter += int((yt_fg & yp_fg).sum())
        union += int((yt_fg | yp_fg).sum())
    return inter / union if union > 0 else 0.0


def format_per_class_block(title, stats):
    lines = [title]
    for s in stats:
        lines.append(
            "  {cid:>2d} ({name:<20s}) acc={acc:.3f}  {correct}/{count}".format(
                cid=s["class_id"],
                name=s["class_name"],
                acc=s["acc"],
                correct=s["correct"],
                count=s["count"],
            )
        )
    return "\n".join(lines)


def format_balance_block(title, counter, class_names):
    total = sum(counter.values())
    lines = ["{title} (total={total})".format(title=title, total=total)]
    for i, name in enumerate(class_names):
        cnt = counter.get(i, 0)
        ratio = cnt / total if total > 0 else 0.0
        lines.append(
            "  {cid:>2d} ({name:<20s}) count={cnt:<8d} ratio={ratio:.4f}".format(
                cid=i,
                name=name,
                cnt=cnt,
                ratio=ratio,
            )
        )
    return "\n".join(lines)


def format_seg_iou_block(title, stats):
    lines = [title]
    for s in stats:
        lines.append(
            "  {cid:>2d} ({name:<20s}) iou={iou:.3f}  inter={inter} union={union}".format(
                cid=s["class_id"],
                name=s["class_name"],
                iou=s["iou"],
                inter=s["intersection"],
                union=s["union"],
            )
        )
    return "\n".join(lines)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_text(text, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(text)


def torch_argmax_list(logits):
    return logits.argmax(dim=1).detach().cpu().tolist()
