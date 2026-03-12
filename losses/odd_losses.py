import torch
import torch.nn.functional as F

from configs.odd_config import ODD_HEAD_CONFIG

def odd_loss(outputs, batch):
    device = next(iter(outputs.values())).device
    total = 0.0

    for name, cfg in ODD_HEAD_CONFIG.items():
        if name not in outputs or name not in batch["labels"]:
            continue

        logits = outputs[name]
        labels = batch["labels"][name].to(device)
        mask = batch["mask"][name].float().to(device)
        sev = batch["severity"][name].float().to(device)
        w = cfg.get("loss_weight", 1.0)

        weight = mask * sev  # [B]
        if weight.sum() <= 0:
            continue

        # --- BULLETPROOF FIX ---
        n_classes = logits.shape[1]
        
        # 1. Create a mask of strictly valid labels (e.g. 0 to 5)
        # This flags any -1, 255, etc. as False
        valid_labels_mask = (labels >= 0) & (labels < n_classes)
        
        # If no valid labels exist in this batch for this head, skip to prevent div by zero
        if not valid_labels_mask.any():
            continue

        # 2. Clamp labels to a safe range so the CUDA kernel doesn't crash.
        # (The bad labels will be completely ignored by the mask in step 3 anyway)
        safe_labels = labels.clamp(min=0, max=n_classes - 1)

        # Calculate raw loss
        per = F.cross_entropy(logits, safe_labels, reduction="none")
        
        # 3. Combine the severity/missing mask with our new valid labels mask
        final_weight = weight * valid_labels_mask.float()
        
        # 4. Final calculation
        loss_head = (per * final_weight).sum() / (final_weight.sum() + 1e-6)

        total = total + w * loss_head

    return total