import os
import torch


def load_checkpoint(path: str, map_location: str = "cpu"):
    return torch.load(path, map_location=map_location)


def extract_model_state(ckpt):
    return ckpt.get("model", ckpt)


def load_model_ckpt(model, ckpt_path: str, map_location: str = "cpu"):
    ckpt = load_checkpoint(ckpt_path, map_location=map_location)
    state = extract_model_state(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


def save_checkpoint(state: dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)


def save_last_and_best(state: dict, save_dir: str, score: float, best_score: float):
    os.makedirs(save_dir, exist_ok=True)

    last_path = os.path.join(save_dir, "last.pt")
    best_path = os.path.join(save_dir, "best.pt")

    save_checkpoint(state, last_path)

    is_best = score > best_score
    if is_best:
        save_checkpoint(state, best_path)
        best_score = score

    return best_score, is_best