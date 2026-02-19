import os
import sys

# add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
from models.odd_model import ODDModel


def main():
    model = ODDModel(freeze_backbone=True)
    model.eval()

    x = torch.randn(2, 3, 336, 336)

    with torch.no_grad():
        out = model(x)

    print("heads:", list(out.keys()))
    for k, v in out.items():
        print(k, v.shape)


if __name__ == "__main__":
    main()
