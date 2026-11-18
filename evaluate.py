"""Evaluate a trained checkpoint on the UCF101 test split with multi-view testing.

    python evaluate.py --checkpoint checkpoints/<run>/best.pth --num_clips 3

Writes a classification report, a confusion-matrix plot and a JSON summary to ``--output_dir``.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.ucf101 import UCF101Dataset
from utils import autocast_dtype, get_device, load_model_from_checkpoint


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    probs_all, targets_all = [], []
    for clips, targets in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
        clips = clips.to(device, non_blocking=True)
        b, v = clips.shape[:2]
        ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
        with ctx:
            logits = model(clips.flatten(0, 1)).float()
        probs_all.append(logits.softmax(-1).view(b, v, -1).mean(1).cpu())
        targets_all.append(targets)
    return torch.cat(probs_all), torch.cat(targets_all)


def plot_confusion(cm, classes, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(22, 22))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=6)
    ax.set_yticks(range(len(classes)), classes, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("UCF101 confusion matrix (row-normalized)")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Evaluate on UCF101 test split")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="./datasets/ucf101")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--split_mode", default=None, choices=["official", "kaggle"],
                   help="Defaults to the split mode the checkpoint was trained with")
    p.add_argument("--num_clips", type=int, default=3, help="Temporal views per video (softmax averaged)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only N videos (quick check)")
    args = p.parse_args()

    device = get_device(args.device)
    amp_dtype = autocast_dtype(device)
    model, classes, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    name = ckpt["model_name"]
    print(f"Loaded {name} (epoch {ckpt['epoch']}, val metrics {ckpt.get('metrics')})")

    split_mode = args.split_mode or ckpt.get("args", {}).get("split_mode", "official")
    ds = UCF101Dataset(args.data_root, args.split, model.cfg, classes=classes, num_clips=args.num_clips,
                       train=False, split_mode=split_mode)
    if args.limit:  # evenly spaced subset, for quick checks
        ds.samples = ds.samples[:: max(len(ds.samples) // args.limit, 1)][: args.limit]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    print(f"{args.split} ({split_mode}): {len(ds)} videos x {args.num_clips} views")

    probs, targets = predict(model, loader, device, amp_dtype)
    preds = probs.argmax(1)
    top1 = (preds == targets).float().mean().item() * 100
    top5 = (probs.topk(5, 1).indices == targets[:, None]).any(1).float().mean().item() * 100
    print(f"{args.split} top-1: {top1:.2f}%  top-5: {top5:.2f}%")

    os.makedirs(args.output_dir, exist_ok=True)
    y, yhat = targets.numpy(), preds.numpy()
    labels = list(range(len(classes)))
    cm = confusion_matrix(y, yhat, labels=labels)
    report = classification_report(y, yhat, labels=labels, target_names=classes, digits=3, zero_division=0)
    np.save(os.path.join(args.output_dir, f"{name}_confusion_matrix.npy"), cm)
    plot_confusion(cm, classes, os.path.join(args.output_dir, f"{name}_confusion_matrix.png"))
    with open(os.path.join(args.output_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(f"Model: {name}\nCheckpoint: {args.checkpoint}\nSplit: {args.split}/{split_mode} ({len(ds)} videos, "
                f"{args.num_clips} views)\nTop-1: {top1:.2f}%\nTop-5: {top5:.2f}%\n\n{report}")
    per_class = cm.diagonal() / cm.sum(1).clip(min=1)
    worst = np.argsort(per_class)[:10]
    summary = {
        "model": name, "checkpoint": args.checkpoint, "split": args.split, "split_mode": split_mode,
        "num_videos": len(ds),
        "num_clips": args.num_clips, "top1": top1, "top5": top5,
        "worst_classes": {classes[i]: float(per_class[i]) for i in worst},
    }
    with open(os.path.join(args.output_dir, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
