import torch
from torch.utils.data import DataLoader, random_split

from data.bdd_dataset import BDDDTimeScene, BDDDVisibility, collate_time_scene
from data.rscd_dataset import RSCDRoadCondition


ALL_HEADS = ["time", "scene", "visibility", "road_condition"]


def collate_visibility(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    labels = torch.stack([x[1] for x in batch], dim=0)
    bsz = labels.shape[0]
    return {
        "images": imgs,
        "labels": {
            "visibility": labels,
        },
        "mask": {
            "visibility": torch.ones(bsz, dtype=torch.float32),
        },
        "severity": {
            "visibility": torch.ones(bsz, dtype=torch.float32),
        },
    }


def collate_road_condition(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    labels = torch.stack([x[1] for x in batch], dim=0)
    bsz = labels.shape[0]
    return {
        "images": imgs,
        "labels": {
            "road_condition": labels,
        },
        "mask": {
            "road_condition": torch.ones(bsz, dtype=torch.float32),
        },
        "severity": {
            "road_condition": torch.ones(bsz, dtype=torch.float32),
        },
    }


def make_loader(dataset, batch_size, shuffle, num_workers, collate_fn=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def move_batch_to_device(batch, device):
    batch["images"] = batch["images"].to(device, non_blocking=True)
    for group in ["labels", "mask", "severity"]:
        for k in batch[group]:
            batch[group][k] = batch[group][k].to(device, non_blocking=True)
    return batch


def fill_missing_heads(batch):
    device = batch["images"].device
    bsz = batch["images"].shape[0]

    for name in ALL_HEADS:
        if name not in batch["labels"]:
            batch["labels"][name] = torch.zeros(bsz, dtype=torch.long, device=device)
        if name not in batch["mask"]:
            batch["mask"][name] = torch.zeros(bsz, dtype=torch.float32, device=device)
        if name not in batch["severity"]:
            batch["severity"][name] = torch.ones(bsz, dtype=torch.float32, device=device)

    return batch


def build_multitask_datasets(bdd_root, rscd_root, seed=42, road_val_ratio=0.1):
    ts_train = BDDDTimeScene(
        img_root=f"{bdd_root}/100k_datasets/100k/train",
        label_dir=f"{bdd_root}/100k_label/100k/train",
    )
    ts_val = BDDDTimeScene(
        img_root=f"{bdd_root}/100k_datasets/100k/val",
        label_dir=f"{bdd_root}/100k_label/100k/val",
    )

    vis_train = BDDDVisibility(split="train")
    vis_val = BDDDVisibility(split="val")

    road_full = RSCDRoadCondition(root=rscd_root, split="train")
    n = len(road_full)
    val_len = max(1, int(road_val_ratio * n))
    train_len = n - val_len
    gen = torch.Generator().manual_seed(seed)
    road_train, road_val = random_split(road_full, [train_len, val_len], generator=gen)

    return {
        "ts_train": ts_train,
        "ts_val": ts_val,
        "vis_train": vis_train,
        "vis_val": vis_val,
        "road_train": road_train,
        "road_val": road_val,
    }


def build_multitask_loaders(datasets, batch_size, num_workers):
    train_loader_ts = make_loader(
        datasets["ts_train"], batch_size, True, num_workers, collate_time_scene
    )
    val_loader_ts = make_loader(
        datasets["ts_val"], batch_size, False, num_workers, collate_time_scene
    )

    train_loader_vis = make_loader(
        datasets["vis_train"], batch_size, True, num_workers, collate_visibility
    )
    val_loader_vis = make_loader(
        datasets["vis_val"], batch_size, False, num_workers, collate_visibility
    )

    train_loader_road = make_loader(
        datasets["road_train"], batch_size, True, num_workers, collate_road_condition
    )
    val_loader_road = make_loader(
        datasets["road_val"], batch_size, False, num_workers, collate_road_condition
    )

    return {
        "train_ts": train_loader_ts,
        "val_ts": val_loader_ts,
        "train_vis": train_loader_vis,
        "val_vis": val_loader_vis,
        "train_road": train_loader_road,
        "val_road": val_loader_road,
    }