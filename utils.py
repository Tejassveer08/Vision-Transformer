"""Shared helpers: seeding, metrics, checkpoint I/O."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "cuda") -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


def autocast_dtype(device: torch.device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@torch.no_grad()
def topk_correct(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5)):
    maxk = min(max(ks), logits.shape[1])
    pred = logits.topk(maxk, dim=1).indices
    hit = pred.eq(targets.view(-1, 1))
    return [hit[:, : min(k, maxk)].any(dim=1).sum().item() for k in ks]


def save_checkpoint(path: str, model, classes, epoch: int, metrics: dict, args: dict | None = None, extra: dict | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "model_name": model.model_name,
        "model_cfg": model.cfg.to_dict(),
        "classes": list(classes),
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": args or {},
    }
    if extra:
        state.update(extra)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_model_from_checkpoint(path: str, device="cpu"):
    """Rebuild the model from a checkpoint. Returns ``(model, classes, checkpoint)``."""
    from models.video_models import VideoModelConfig, get_model

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = VideoModelConfig.from_dict(ckpt["model_cfg"])
    model = get_model(ckpt["model_name"], num_classes=len(ckpt["classes"]), pretrained=False, cfg=cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["classes"], ckpt
