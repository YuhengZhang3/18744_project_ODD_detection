import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleHead(nn.Module):
    # cls feature ->logits

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
    # patch features ->attention pooling ->logits

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