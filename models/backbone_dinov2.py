import torch
import torch.nn as nn

from configs.odd_config import BACKBONE_NAME


class DinoBackbone(nn.Module):
    # dinov2 vit backbone only
    # adapters are moved to odd_model.py

    def __init__(self, backbone_name=BACKBONE_NAME):
        super().__init__()

        self.vit = torch.hub.load(
            "facebookresearch/dinov2",
            backbone_name,
            pretrained=True,
        )

        self.feat_dim = getattr(self.vit, "embed_dim", None)
        if self.feat_dim is None:
            raise RuntimeError("cannot infer DINO feature dim")

    @torch.no_grad()
    def freeze_backbone(self):
        for p in self.vit.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x: [B, 3, H, W]
        feats = self.vit.forward_features(x)

        if isinstance(feats, dict):
            if "x_norm_clstoken" in feats and "x_norm_patchtokens" in feats:
                cls_token = feats["x_norm_clstoken"]      # [B, D]
                patch_tokens = feats["x_norm_patchtokens"]  # [B, N, D]
            elif "x" in feats:
                tokens = feats["x"]
                cls_token = tokens[:, 0]
                patch_tokens = tokens[:, 1:]
            else:
                cls_token = feats[:, 0]
                patch_tokens = feats[:, 1:]
        else:
            cls_token = feats[:, 0]
            patch_tokens = feats[:, 1:]

        return cls_token, patch_tokens