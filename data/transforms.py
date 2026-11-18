"""Custom video transforms.

Written in the style of the PyTorch "custom dataset / transforms / loader" recipe:
every transform is a small callable class, composed with ``Compose``.

All transforms after ``ToTensorVideo`` operate on a float tensor shaped
``[T, C, H, W]``; ``ToCTHW`` is the last step and produces the ``[C, T, H, W]``
layout that 3D video models expect.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip

    def __repr__(self):
        lines = ",\n    ".join(repr(t) for t in self.transforms)
        return f"{self.__class__.__name__}(\n    {lines}\n)"


class ToTensorVideo:
    """uint8 numpy ``[T, H, W, C]`` -> float tensor ``[T, C, H, W]`` in ``[0, 1]``."""

    def __call__(self, clip: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(clip))
        return tensor.permute(0, 3, 1, 2).float().div_(255.0)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ShortSideResize:
    """Resize so the short side equals ``size`` (keeps aspect ratio)."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return _resize_short_side(clip, self.size)

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class RandomShortSideResize:
    """Scale jitter: resize the short side to a random value in ``[min_size, max_size]``."""

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return _resize_short_side(clip, random.randint(self.min_size, self.max_size))

    def __repr__(self):
        return f"{self.__class__.__name__}(min_size={self.min_size}, max_size={self.max_size})"


class RandomCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        h, w = clip.shape[-2:]
        top = random.randint(0, max(h - self.size, 0))
        left = random.randint(0, max(w - self.size, 0))
        return clip[..., top : top + self.size, left : left + self.size]

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class CenterCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        h, w = clip.shape[-2:]
        top = max((h - self.size) // 2, 0)
        left = max((w - self.size) // 2, 0)
        return clip[..., top : top + self.size, left : left + self.size]

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return clip.flip(-1)
        return clip

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class ColorJitterVideo:
    """Brightness / contrast jitter applied identically to every frame of the clip."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, p: float = 0.8):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return clip
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean = clip.mean(dim=(-3, -2, -1), keepdim=True)
        return ((clip * b - mean) * c + mean).clamp_(0.0, 1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(brightness={self.brightness}, contrast={self.contrast}, p={self.p})"


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return (clip - self.mean) / self.std

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean.flatten().tolist()}, std={self.std.flatten().tolist()})"


class ToCTHW:
    """``[T, C, H, W]`` -> ``[C, T, H, W]``."""

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return clip.permute(1, 0, 2, 3).contiguous()

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def _resize_short_side(clip: torch.Tensor, size: int) -> torch.Tensor:
    h, w = clip.shape[-2:]
    if h <= w:
        new_h, new_w = size, int(round(w * size / h))
    else:
        new_h, new_w = int(round(h * size / w)), size
    if (new_h, new_w) == (h, w):
        return clip
    return F.interpolate(clip, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)


def build_transforms(cfg, train: bool) -> Compose:
    """Build the train / eval pipeline for a model config (see ``models.video_models.VideoModelConfig``)."""
    if train:
        return Compose(
            [
                ToTensorVideo(),
                RandomShortSideResize(cfg.crop_size, int(cfg.resize_size * 1.15)),
                RandomCrop(cfg.crop_size),
                RandomHorizontalFlip(0.5),
                ColorJitterVideo(),
                Normalize(cfg.mean, cfg.std),
                ToCTHW(),
            ]
        )
    return Compose(
        [
            ToTensorVideo(),
            ShortSideResize(cfg.resize_size),
            CenterCrop(cfg.crop_size),
            Normalize(cfg.mean, cfg.std),
            ToCTHW(),
        ]
    )
