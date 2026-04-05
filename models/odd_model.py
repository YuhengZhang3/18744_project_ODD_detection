import torch.nn as nn

from configs.odd_config import ODD_HEAD_CONFIG, ADAPTER_DIM
from .backbone_dinov2 import DinoBackbone
from .heads import SimpleHead, PatchSegHead


_BDD_GLOBAL_HEADS = {"time", "scene", "visibility", "anomalies"}
_ROAD_HEADS = {"road_condition", "road_state", "road_severity"}
_DENSE_HEADS = {"drivable"}


def make_adapter(in_dim, out_dim):
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
    )


class ODDModel(nn.Module):
    # shared DINO backbone
    # bdd tasks use shared bdd adapter
    # road task uses private road adapter

    def __init__(self, freeze_backbone=False):
        super().__init__()

        self.backbone = DinoBackbone()
        if freeze_backbone:
            self.backbone.freeze_backbone()

        feat_dim = self.backbone.feat_dim

        # shared adapter for bdd heads
        self.bdd_adapter = make_adapter(feat_dim, ADAPTER_DIM)

        # private adapter for road branch
        self.road_adapter = make_adapter(feat_dim, ADAPTER_DIM)

        # private adapter for anomaly branch
        self.anomaly_adapter = make_adapter(feat_dim, ADAPTER_DIM)

        self.heads = nn.ModuleDict()

        for name, cfg in ODD_HEAD_CONFIG.items():
            num_classes = cfg["num_classes"]

            if name in _BDD_GLOBAL_HEADS:
                self.heads[name] = SimpleHead(ADAPTER_DIM, num_classes)

            elif name in _ROAD_HEADS:
                self.heads[name] = SimpleHead(ADAPTER_DIM, num_classes)

            elif name in _DENSE_HEADS:
                self.heads[name] = PatchSegHead(
                    ADAPTER_DIM,
                    num_classes,
                    upsample_size=336,
                )

            else:
                raise ValueError(f"unknown head name: {name}")

    def forward(self, x):
        raw_cls, raw_patch = self.backbone(x)

        # bdd branch
        bdd_cls = self.bdd_adapter(raw_cls)
        bdd_patch = self.bdd_adapter(raw_patch)

        # road branch
        road_cls = self.road_adapter(raw_cls)

        # anomaly branch
        anomaly_cls = self.anomaly_adapter(raw_cls)

        out = {}

        # bdd global heads
        for name in _BDD_GLOBAL_HEADS:
            if name in self.heads:
                if name == "anomalies":
                    out[name] = self.heads[name](anomaly_cls)
                else:
                    out[name] = self.heads[name](bdd_cls)

        # road head
        for name in _ROAD_HEADS:
            if name in self.heads:
                out[name] = self.heads[name](road_cls)

        # dense head
        for name in _DENSE_HEADS:
            if name in self.heads:
                out[name] = self.heads[name](bdd_patch)

        return out