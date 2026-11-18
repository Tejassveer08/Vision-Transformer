"""Run action recognition on a single video.

    python inference.py --checkpoint checkpoints/<run>/best.pth --video path/to/clip.mp4

Also importable: ``ActionRecognizer(checkpoint).predict(video_path)``. The web app uses it.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch

from data.transforms import build_transforms
from data.ucf101 import load_clip
from utils import autocast_dtype, get_device, load_model_from_checkpoint


class ActionRecognizer:
    def __init__(self, checkpoint: str, device: str = "cuda", num_clips: int = 3):
        self.device = get_device(device)
        self.amp_dtype = autocast_dtype(self.device)
        self.model, self.classes, ckpt = load_model_from_checkpoint(checkpoint, self.device)
        self.model_name = ckpt["model_name"]
        self.metrics = ckpt.get("metrics", {})
        self.cfg = self.model.cfg
        self.transform = build_transforms(self.cfg, train=False)
        self.num_clips = num_clips

    @torch.no_grad()
    def predict(self, video_path: str, top_k: int = 5, num_clips: int | None = None):
        """Return ``[(class_name, probability), ...]``, highest first."""
        v = num_clips or self.num_clips
        clips = torch.stack([self.transform(load_clip(video_path, self.cfg, False, k, v)) for k in range(v)])
        clips = clips.to(self.device)
        ctx = torch.autocast(self.device.type, dtype=self.amp_dtype) if self.amp_dtype else nullcontext()
        with ctx:
            logits = self.model(clips).float()
        probs = logits.softmax(-1).mean(0)
        p, i = probs.topk(min(top_k, len(self.classes)))
        return [(self.classes[j], float(q)) for q, j in zip(p.tolist(), i.tolist())]


def main():
    parser = argparse.ArgumentParser(description="Action recognition inference on one video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_clips", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    rec = ActionRecognizer(args.checkpoint, args.device, args.num_clips)
    print(f"Model: {rec.model_name} on {rec.device}")
    for rank, (name, prob) in enumerate(rec.predict(args.video, args.top_k), 1):
        print(f"{rank}. {name:<24s} {prob * 100:6.2f}%")


if __name__ == "__main__":
    main()
