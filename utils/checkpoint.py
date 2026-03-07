import os
import torch


def load_checkpoint(path: str, map_location: str = "cpu"):
    ckpt = torch.load(path, map_location=map_location)
    return ckpt


def extract_model_state(ckpt):
    return ckpt.get("model", ckpt)


def save_checkpoint(state: dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)