import torch
import torch.nn.functional as F

from configs.odd_config import ODD_HEAD_CONFIG


def odd_loss(outputs, batch):
    # outputs: dict(name -> logits)
    # batch: dict with "labels", "mask", "severity"
    device = next(iter(outputs.values())).device
    total = 0.0

    for name, cfg in ODD_HEAD_CONFIG.items():
        if name not in outputs:
            continue

        logits = outputs[name]
        labels = batch["labels"][name].to(device)
        mask = batch["mask"][name].float().to(device)
        sev = batch["severity"][name].float().to(device)
        w = cfg.get("loss_weight", 1.0)

        weight = mask * sev  # [B]
        if weight.sum() <= 0:
            continue

        per = F.cross_entropy(logits, labels, reduction="none")
        per = per * weight
        loss_head = per.sum() / (weight.sum() + 1e-6)

        total = total + w * loss_head

    return total