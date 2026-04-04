import os
import json
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from data.bdd_dataset import (
    BDDDTimeScene,
    BDDDVisibility,
    BDDDDrivable,
    collate_time_scene,
    collate_drivable,
    get_bdd_root,
    resolve_img_root_for_split,
    resolve_label_root_for_split,
)
from data.rscd_dataset import RSCDRoadCondition


ALL_HEADS = ["time", "scene", "visibility", "road_condition", "road_state", "road_severity", "drivable"]


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
    road_condition = torch.stack([x[1]["road_condition"] for x in batch], dim=0)
    road_state = torch.stack([x[1]["road_state"] for x in batch], dim=0)
    road_severity = torch.stack([x[1]["road_severity"] for x in batch], dim=0)
    bsz = road_condition.shape[0]

    return {
        "images": imgs,
        "labels": {
            "road_condition": road_condition,
            "road_state": road_state,
            "road_severity": road_severity,
        },
        "mask": {
            "road_condition": torch.ones(bsz, dtype=torch.float32),
            "road_state": torch.ones(bsz, dtype=torch.float32),
            "road_severity": torch.ones(bsz, dtype=torch.float32),
        },
        "severity": {
            "road_condition": torch.ones(bsz, dtype=torch.float32),
            "road_state": torch.ones(bsz, dtype=torch.float32),
            "road_severity": torch.ones(bsz, dtype=torch.float32),
        },
    }


def make_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    collate_fn=None,
    sampler=None,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
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
            if name == "drivable":
                batch["labels"][name] = torch.zeros(
                    (bsz, 336, 336), dtype=torch.long, device=device
                )
            else:
                batch["labels"][name] = torch.zeros(
                    bsz, dtype=torch.long, device=device
                )

        if name not in batch["mask"]:
            batch["mask"][name] = torch.zeros(
                bsz, dtype=torch.float32, device=device
            )

        if name not in batch["severity"]:
            batch["severity"][name] = torch.ones(
                bsz, dtype=torch.float32, device=device
            )

    return batch


def counter_to_class_weights(counter, num_classes, min_count=1):
    total = sum(counter.values())
    weights = []
    for i in range(num_classes):
        cnt = max(counter.get(i, 0), min_count)
        w = total / cnt
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.mean()
    return weights


def build_sample_weights_from_targets(targets, num_classes):
    counter = {}
    for y in targets:
        y = int(y)
        counter[y] = counter.get(y, 0) + 1

    class_weights = counter_to_class_weights(counter, num_classes)
    sample_weights = torch.tensor(
        [class_weights[int(y)].item() for y in targets],
        dtype=torch.double,
    )
    return sample_weights, class_weights



def load_cached_sample_weights(cache_path):
    return torch.load(cache_path, map_location="cpu")


def maybe_make_weighted_sampler_from_weight_cache(cache_path, enable, subset_indices=None):
    if not enable:
        return None

    weights = load_cached_sample_weights(cache_path)

    if subset_indices is not None:
        weights = weights[subset_indices]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    return sampler

def load_cached_targets(cache_path):
    with open(cache_path, "r") as f:
        return json.load(f)


def maybe_make_weighted_sampler_from_cache(cache_path, num_classes, enable):
    if not enable:
        return None
    targets = load_cached_targets(cache_path)
    sample_weights, _ = build_sample_weights_from_targets(targets, num_classes)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def build_v2_datasets(bdd_root, rscd_root, seed=42, road_val_ratio=0.1):
    if bdd_root is None:
        bdd_root = get_bdd_root()

    ts_train = BDDDTimeScene(
        img_root=resolve_img_root_for_split(bdd_root, "train"),
        label_dir=resolve_label_root_for_split(bdd_root, "train"),
    )
    ts_val = BDDDTimeScene(
        img_root=resolve_img_root_for_split(bdd_root, "val"),
        label_dir=resolve_label_root_for_split(bdd_root, "val"),
    )

    vis_train = BDDDVisibility(split="train")
    vis_val = BDDDVisibility(split="val")

    drv_train = BDDDDrivable(split="train")
    drv_val = BDDDDrivable(split="val")

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
        "drv_train": drv_train,
        "drv_val": drv_val,
        "road_train": road_train,
        "road_val": road_val,
    }


def build_v2_loaders(
    datasets,
    batch_size,
    num_workers,
    use_balanced_scene=False,
    use_balanced_visibility=False,
    use_balanced_road=False,
    sampler_cache_dir="",
    sample_weight_cache_dir="",
    eval_batch_size=0,
):
    scene_sampler = None
    vis_sampler = None
    road_sampler = None

    if sample_weight_cache_dir:
        if use_balanced_scene:
            scene_sampler = maybe_make_weighted_sampler_from_weight_cache(
                os.path.join(sample_weight_cache_dir, "bdd_train_scene_sample_weights.pt"),
                enable=True,
            )

        if use_balanced_visibility:
            vis_sampler = maybe_make_weighted_sampler_from_weight_cache(
                os.path.join(sample_weight_cache_dir, "bdd_train_visibility_sample_weights.pt"),
                enable=True,
            )

        if use_balanced_road:
            subset_indices = datasets["road_train"].indices if hasattr(datasets["road_train"], "indices") else None
            road_sampler = maybe_make_weighted_sampler_from_weight_cache(
                os.path.join(sample_weight_cache_dir, "rscd_train_sample_weights.pt"),
                enable=True,
                subset_indices=subset_indices,
            )

    elif sampler_cache_dir:
        if use_balanced_scene:
            scene_sampler = maybe_make_weighted_sampler_from_cache(
                os.path.join(sampler_cache_dir, "bdd_train_scene_targets.json"),
                num_classes=7,
                enable=True,
            )

        if use_balanced_visibility:
            vis_sampler = maybe_make_weighted_sampler_from_cache(
                os.path.join(sampler_cache_dir, "bdd_train_visibility_targets.json"),
                num_classes=3,
                enable=True,
            )

        if use_balanced_road:
            full_targets = load_cached_targets(
                os.path.join(sampler_cache_dir, "rscd_train_targets.json")
            )

            if hasattr(datasets["road_train"], "indices"):
                road_subset_targets = [full_targets[i] for i in datasets["road_train"].indices]
            else:
                road_subset_targets = full_targets

            sample_weights, _ = build_sample_weights_from_targets(
                road_subset_targets,
                num_classes=27,
            )
            road_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )

    val_bs = eval_batch_size if eval_batch_size and eval_batch_size > 0 else batch_size

    train_loader_ts = make_loader(
        datasets["ts_train"],
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_time_scene,
        sampler=scene_sampler,
    )
    val_loader_ts = make_loader(
        datasets["ts_val"],
        val_bs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_time_scene,
    )

    train_loader_vis = make_loader(
        datasets["vis_train"],
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_visibility,
        sampler=vis_sampler,
    )
    val_loader_vis = make_loader(
        datasets["vis_val"],
        val_bs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_visibility,
    )

    train_loader_drv = make_loader(
        datasets["drv_train"],
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_drivable,
    )
    val_loader_drv = make_loader(
        datasets["drv_val"],
        val_bs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_drivable,
    )

    train_loader_road = make_loader(
        datasets["road_train"],
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_road_condition,
        sampler=road_sampler,
    )
    val_loader_road = make_loader(
        datasets["road_val"],
        val_bs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_road_condition,
    )

    return {
        "train_ts": train_loader_ts,
        "val_ts": val_loader_ts,
        "train_vis": train_loader_vis,
        "val_vis": val_loader_vis,
        "train_drv": train_loader_drv,
        "val_drv": val_loader_drv,
        "train_road": train_loader_road,
        "val_road": val_loader_road,
    }
