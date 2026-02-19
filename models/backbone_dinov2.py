import torch
import torch.nn as nn

from configs.odd_config import BACKBONE_NAME, ADAPTER_DIM


class DinoBackbone(nn.Module):
    # dinov2 vit + small adapter

    def __init__(self, backbone_name=BACKBONE_NAME, out_dim=ADAPTER_DIM):
        super().__init__()

        self.vit = torch.hub.load(
            "facebookresearch/dinov2",
            backbone_name,
            pretrained=True,
        )

        d_model = getattr(self.vit, "embed_dim", out_dim)

        self.adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
            nn.GELU(),
        )

        self.out_dim = out_dim

    @torch.no_grad()
    def freeze_backbone(self):
        for p in self.vit.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x: [B, 3, H, W]
        feats = self.vit.forward_features(x)

        if isinstance(feats, dict):
            if "x_norm_clstoken" in feats and "x_norm_patchtokens" in feats:
                cls_token = feats["x_norm_clstoken"]
                patch_tokens = feats["x_norm_patchtokens"]
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

        cls_feat = self.adapter(cls_token)
        patch_feat = self.adapter(patch_tokens)

        return cls_feat, patch_feat