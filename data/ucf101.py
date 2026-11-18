"""UCF101 dataset (Kaggle ``matthewjansen/ucf101-action-recognition`` layout).

Expected layout after download::

    datasets/ucf101/
        train/<ClassName>/*.avi
        val/<ClassName>/*.avi
        test/<ClassName>/*.avi

The class list is the sorted folder names under ``train/``. The same mapping is
stored in every checkpoint, so inference does not need the dataset.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.transforms import build_transforms

VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov", ".webm"}
KAGGLE_DATASET = "matthewjansen/ucf101-action-recognition"

# OpenCV spawns its own threads per worker; with many DataLoader workers that oversubscribes the CPU.
cv2.setNumThreads(1)


def download_ucf101(root: str = "./datasets/ucf101"):
    """Download and unzip the Kaggle dataset (needs ~/.kaggle/kaggle.json)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    os.makedirs(root, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} to {root} ...")
    api.dataset_download_files(KAGGLE_DATASET, path=root, unzip=True, quiet=False)


def find_classes(root: str) -> list[str]:
    train_dir = Path(root) / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"{train_dir} not found. Run `python -m data.ucf101 --download` or pass --data_root."
        )
    return sorted(d.name for d in train_dir.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------- splits
# UCF101 clips come in 25 groups per class; clips within a group share actor and background.
# The official split 1 tests on groups 1-7. The Kaggle folders split randomly per clip, so
# nearly every test group also appears in train and test accuracy is inflated.
# "official" pools all Kaggle folders and re-splits by group:
#   test = g01-g07 (official split-1 test), val = g08-g10, train = g11-g25.
OFFICIAL_GROUPS = {"test": range(1, 8), "val": range(8, 11), "train": range(11, 26)}
_GROUP_RE = re.compile(r"_g(\d+)_c\d+", re.IGNORECASE)


def _scan_folder(split_dir: Path):
    out = []
    if not split_dir.is_dir():
        return out
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in VIDEO_EXTS:
                out.append((str(f), cls_dir.name))
    return out


def list_samples(root: str, split: str, split_mode: str = "official") -> list[tuple[str, str]]:
    """Return ``[(video_path, class_name), ...]`` for a split."""
    root = Path(root)
    if split_mode == "kaggle":
        if not (root / split).is_dir():
            raise FileNotFoundError(f"Split folder not found: {root / split}")
        return _scan_folder(root / split)
    if split_mode != "official":
        raise ValueError(f"split_mode must be 'official' or 'kaggle', got {split_mode}")
    groups = set(OFFICIAL_GROUPS[split])
    out = []
    for folder in ("train", "val", "test"):
        for path, cls in _scan_folder(root / folder):
            m = _GROUP_RE.search(Path(path).stem)
            if m and int(m.group(1)) in groups:
                out.append((path, cls))
    return sorted(out)


# ---------------------------------------------------------------------------- video io
def get_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 0)


def sample_indices(num_total: int, cfg, train: bool, clip_idx: int = 0, num_clips: int = 1) -> np.ndarray:
    """Choose which frame indices to decode.

    ``clip_idx`` / ``num_clips`` give evenly spaced views across the video for multi-view testing.
    """
    n = max(num_total, 1)
    t = cfg.num_frames
    if cfg.sampling == "segment":
        edges = np.linspace(0, n, t + 1)
        if train:
            idx = [random.uniform(edges[i], edges[i + 1]) for i in range(t)]
        else:
            # Each view shifts the sampled position within every segment.
            frac = (clip_idx + 0.5) / num_clips
            idx = edges[:-1] + frac * (edges[1:] - edges[:-1])
        return np.clip(np.asarray(idx).astype(int), 0, n - 1)

    # dense
    span = (t - 1) * cfg.sampling_rate + 1
    max_start = max(n - span, 0)
    if train:
        start = random.randint(0, max_start)
    elif num_clips == 1:
        start = max_start // 2
    else:
        start = int(round(max_start * clip_idx / (num_clips - 1)))
    idx = start + np.arange(t) * cfg.sampling_rate
    # Videos shorter than the span: loop instead of repeating the last frame forever.
    return np.mod(idx, n)


def read_frames(path: str, indices: np.ndarray) -> np.ndarray:
    """Decode the requested frame indices (RGB uint8, ``[T, H, W, 3]``) in one sequential pass."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video {path}")
    wanted = set(int(i) for i in indices)
    last = max(wanted)
    frames: dict[int, np.ndarray] = {}
    last_good = None
    i = 0
    while i <= last:
        ok = cap.grab()
        if not ok:
            break
        if i in wanted:
            ok, frame = cap.retrieve()
            if ok:
                last_good = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[i] = last_good
        i += 1
    cap.release()
    if last_good is None:
        raise IOError(f"Could not decode any frames from {path}")
    # Frame counts in AVI headers are sometimes too high: fill missing tail frames with the nearest decoded one.
    decoded = sorted(frames)
    out = []
    for j in indices:
        j = int(j)
        if j in frames:
            out.append(frames[j])
        else:
            nearest = min(decoded, key=lambda k: abs(k - j))
            out.append(frames[nearest])
    return np.stack(out)


def load_clip(path: str, cfg, train: bool, clip_idx: int = 0, num_clips: int = 1) -> np.ndarray:
    n = get_frame_count(path)
    return read_frames(path, sample_indices(n, cfg, train, clip_idx, num_clips))


# ---------------------------------------------------------------------------- dataset
class UCF101Dataset(Dataset):
    """Returns ``(clip [C, T, H, W] float tensor, label int)``.

    With ``num_clips > 1`` (evaluation), returns ``[num_clips, C, T, H, W]`` covering the whole video.
    """

    def __init__(self, root: str, split: str, cfg, transform=None, classes: list[str] | None = None,
                 num_clips: int = 1, train: bool | None = None, split_mode: str = "official"):
        self.root = Path(root)
        self.split = split
        self.cfg = cfg
        self.train = (split == "train") if train is None else train
        self.transform = transform or build_transforms(cfg, self.train)
        self.num_clips = num_clips
        self.classes = classes or find_classes(root)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = [(p, self.class_to_idx[c]) for p, c in list_samples(root, split, split_mode)
                        if c in self.class_to_idx]
        if not self.samples:
            raise RuntimeError(f"No videos found for split '{split}' ({split_mode}) under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            if self.num_clips == 1:
                clip = self.transform(load_clip(path, self.cfg, self.train))
            else:
                clip = torch.stack([
                    self.transform(load_clip(path, self.cfg, False, k, self.num_clips))
                    for k in range(self.num_clips)
                ])
        except IOError as e:
            # A corrupt video should not kill a multi-hour run; substitute a random other sample.
            print(f"[warn] {e}; substituting another sample")
            return self[random.randrange(len(self))]
        return clip, label


def get_dataloaders(root: str, cfg, batch_size: int = 4, num_workers: int = 4,
                    eval_batch_size: int | None = None, eval_clips: int = 1,
                    splits=("train", "val"), split_mode: str = "official"):
    """Build one DataLoader per split. Returns ``(loaders: dict, classes: list[str])``."""
    classes = find_classes(root)
    loaders = {}
    for split in splits:
        train = split == "train"
        ds = UCF101Dataset(root, split, cfg, classes=classes, num_clips=1 if train else eval_clips,
                           split_mode=split_mode)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size if train else (eval_batch_size or batch_size),
            shuffle=train,
            drop_last=train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    return loaders, classes


if __name__ == "__main__":
    import argparse

    from models.video_models import MODEL_CONFIGS

    parser = argparse.ArgumentParser(description="Download / inspect UCF101")
    parser.add_argument("--data_root", default="./datasets/ucf101")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model", default="vivit_b16")
    parser.add_argument("--split_mode", default="official", choices=["official", "kaggle"])
    args = parser.parse_args()

    if args.download:
        download_ucf101(args.data_root)
    cfg = MODEL_CONFIGS[args.model]
    loaders, classes = get_dataloaders(args.data_root, cfg, batch_size=2, num_workers=0,
                                       splits=("train", "val", "test"), split_mode=args.split_mode)
    print(f"{len(classes)} classes")
    for split, loader in loaders.items():
        print(f"{split}: {len(loader.dataset)} videos")
    clips, labels = next(iter(loaders["train"]))
    print("batch:", tuple(clips.shape), [classes[i] for i in labels])
