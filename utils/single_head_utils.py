import torch
from torch.utils.data import DataLoader


def make_loader(dataset, batch_size, shuffle, num_workers, collate_fn=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )


def freeze_backbone_only(model):
    for p in model.backbone.parameters():
        p.requires_grad = False


def freeze_vit_only(model):
    for p in model.backbone.vit.parameters():
        p.requires_grad = False


def move_batch_to_device(batch, device):
    if isinstance(batch, dict):
        batch["images"] = batch["images"].to(device, non_blocking=True)
        for group in ["labels", "mask", "severity"]:
            if group in batch:
                for k in batch[group]:
                    batch[group][k] = batch[group][k].to(device, non_blocking=True)
        return batch

    if isinstance(batch, (list, tuple)):
        return [x.to(device, non_blocking=True) for x in batch]

    return batch.to(device, non_blocking=True)


def save_last_and_best(
    state: dict,
    save_dir: str,
    score: float,
    best_score: float,
):
    import os
    import torch

    os.makedirs(save_dir, exist_ok=True)

    last_path = os.path.join(save_dir, "last.pt")
    best_path = os.path.join(save_dir, "best.pt")

    torch.save(state, last_path)

    is_best = score > best_score
    if is_best:
        torch.save(state, best_path)
        best_score = score

    return best_score, is_best