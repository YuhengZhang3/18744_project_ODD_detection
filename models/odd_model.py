import torch.nn as nn

from configs.odd_config import ODD_HEAD_CONFIG, ADAPTER_DIM
from .backbone_dinov2 import DinoBackbone
from .heads import SimpleHead, AttentiveHead


# cls heads: mostly global conditions
#_GLOBAL_HEADS = {"weather", "lighting", "time", "traffic", "road_condition", "scene"}
_GLOBAL_HEADS = {"time", "scene"}

class ODDModel(nn.Module):
    # dinov2 backbone + multi-head classifier

    def __init__(self, freeze_backbone=False):
        super().__init__()

        self.backbone = DinoBackbone(out_dim=ADAPTER_DIM)
        if freeze_backbone:
            self.backbone.freeze_backbone()

        self.heads = nn.ModuleDict()

        for name, cfg in ODD_HEAD_CONFIG.items():
            num_classes = cfg["num_classes"]
            if name in _GLOBAL_HEADS:
                self.heads[name] = SimpleHead(ADAPTER_DIM, num_classes)
            else:
                self.heads[name] = AttentiveHead(ADAPTER_DIM, num_classes)

    def forward(self, x):
        cls_feat, patch_feat = self.backbone(x)

        out = {}
        for name, head in self.heads.items():
            if name in _GLOBAL_HEADS:
                out[name] = head(cls_feat)
            else:
                out[name] = head(patch_feat)

        return out