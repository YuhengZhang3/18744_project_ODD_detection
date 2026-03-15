import torch
import torch.nn.functional as F

from configs.odd_config import ODD_HEAD_CONFIG


def _expand_batch_weight(weight, target_ndim):
    # weight: [B] or already dense
    # target_ndim for segmentation labels is usually 3 -> [B, H, W]
    if weight.ndim == target_ndim:
        return weight
    if weight.ndim == 1 and target_ndim == 3:
        return weight[:, None, None]
    return weight


def odd_loss(outputs, batch):
    # outputs: dict(name -> logits)
    # batch: dict with "labels", "mask", "severity"
    device = next(iter(outputs.values())).device
    total = 0.0

    batch_labels = batch.get("labels", {})
    batch_mask = batch.get("mask", {})
    batch_sev = batch.get("severity", {})

    for name, cfg in ODD_HEAD_CONFIG.items():
        if name not in outputs:
            continue
        if name not in batch_labels:
            continue
        if name not in batch_mask:
            continue
        if name not in batch_sev:
            continue

        logits = outputs[name]
        labels = batch_labels[name].to(device)
        mask = batch_mask[name].float().to(device)
        sev = batch_sev[name].float().to(device)
        w = cfg.get("loss_weight", 1.0)

        head_type = cfg.get("type", "multiclass")

        if head_type == "multiclass":
            # logits: [B, C], labels: [B]
            weight = mask * sev
            if weight.sum() <= 0:
                continue

            per = F.cross_entropy(logits, labels, reduction="none")  # [B]
            per = per * weight
            loss_head = per.sum() / (weight.sum() + 1e-6)
            total = total + w * loss_head

        elif head_type == "segmentation":
            # logits: [B, C, H, W], labels: [B, H, W]
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            per = F.cross_entropy(logits, labels.long(), reduction="none")  # [B, H, W]

            weight = mask * sev
            weight = _expand_batch_weight(weight, per.ndim)

            if weight.sum() <= 0:
                continue

            per = per * weight
            loss_head = per.sum() / (weight.sum() + 1e-6)
            total = total + w * loss_head

        else:
            raise ValueError(f"unknown loss type: {head_type}")

    return total