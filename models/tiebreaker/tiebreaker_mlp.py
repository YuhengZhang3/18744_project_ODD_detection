"""
Stage-2 Tiebreaker MLP.

Input:  60-dim feature vector (stage-1 outputs + virtual sensors)
Output: corrected weather/time/scene/anomalies predictions

Output heads:
  - fog:       1-dim sigmoid  (BCE loss)
  - rain:      1-dim sigmoid  (BCE loss)
  - snow:      1-dim sigmoid  (BCE loss)
  - time:      4-dim softmax  (CE loss, ignore_index=-1)
  - scene:     7-dim softmax  (CE loss, ignore_index=-1)
  - anomalies: 4-dim softmax  (CE loss, ignore_index=-1)
"""

import torch
import torch.nn as nn


class TiebreakerMLP(nn.Module):

    def __init__(self, input_dim=60, hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Weather heads (binary)
        self.head_fog = nn.Linear(hidden2, 1)
        self.head_rain = nn.Linear(hidden2, 1)
        self.head_snow = nn.Linear(hidden2, 1)

        # Classification heads
        self.head_time = nn.Linear(hidden2, 4)
        self.head_scene = nn.Linear(hidden2, 7)
        self.head_anomalies = nn.Linear(hidden2, 4)

    def forward(self, x):
        h = self.backbone(x)
        return {
            "fog": self.head_fog(h).squeeze(-1),           # [B]
            "rain": self.head_rain(h).squeeze(-1),          # [B]
            "snow": self.head_snow(h).squeeze(-1),          # [B]
            "time": self.head_time(h),                      # [B, 4]
            "scene": self.head_scene(h),                    # [B, 7]
            "anomalies": self.head_anomalies(h),            # [B, 4]
        }