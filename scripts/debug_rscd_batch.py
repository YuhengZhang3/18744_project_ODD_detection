import os, sys, torch
from torch.utils.data import DataLoader

root = os.path.dirname(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.append(root)

from data.rscd_dataset import RSCDRoadCondition, collate_rscd, RSCD_CLASSES

ds = RSCDRoadCondition(root="/home/yuhengz3@andrew.cmu.edu/rscd/dataset", split="train")
loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_rscd)

batch = next(iter(loader))
print(batch["images"].shape)
print(batch["labels"]["road_condition"])
print([RSCD_CLASSES[i.item()] for i in batch["labels"]["road_condition"]])