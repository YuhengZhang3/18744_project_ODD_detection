import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleHead(nn.Module):
    # cls feature -> logits

    def __init__(self, in_dim, num_classes):
        super().__init__()
        hid = max(in_dim // 2, 256)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hid, num_classes),
        )

    def forward(self, x):
        return self.mlp(x)


class AttentiveHead(nn.Module):
    # patch features -> attention pooling -> logits

    def __init__(self, in_dim, num_classes, num_queries=4):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(num_queries, in_dim))
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape

        q = self.query.unsqueeze(0).expand(B, self.num_queries, D)
        k = x

        attn = torch.matmul(q, k.transpose(1, 2)) / (D ** 0.5)
        attn = F.softmax(attn, dim=-1)

        pooled = torch.matmul(attn, k)
        pooled = pooled.mean(dim=1)

        logits = self.fc(pooled)
        return logits


class PatchSegHead(nn.Module):
    # patch features -> dense logits
    # input: [B, N, D]
    # output: [B, C, H, W]
    # assumes patch tokens form a square grid

    def __init__(self, in_dim, num_classes, upsample_size=336):
        super().__init__()
        hid = max(in_dim // 2, 256)
        self.num_classes = num_classes
        self.upsample_size = upsample_size

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Linear(hid, num_classes),
        )

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        side = int(math.sqrt(N))
        if side * side != N:
            raise RuntimeError(f"patch token count {N} is not a square number")

        logits = self.proj(x)                      # [B, N, C]
        logits = logits.permute(0, 2, 1).contiguous()  # [B, C, N]
        logits = logits.view(B, self.num_classes, side, side)  # [B, C, h, w]

        if self.upsample_size is not None:
            logits = F.interpolate(
                logits,
                size=(self.upsample_size, self.upsample_size),
                mode="bilinear",
                align_corners=False,
            )

        return logits