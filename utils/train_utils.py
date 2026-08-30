import os
import torch


def save_checkpoint(model, optimizer, epoch: int, save_dir: str, filename: str):
    os.makedirs(save_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(state, os.path.join(save_dir, filename))


def load_checkpoint(model, optimizer, ckpt_path: str, map_location="cpu"):
    state = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    epoch = state.get("epoch", 0)
    return model, optimizer, epoch


