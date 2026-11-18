"""Export a slim, deployable checkpoint: fp16 weights and metadata, no optimizer state.

    python export_model.py --checkpoint checkpoints/vivit_b16_ucf101/best.pth \
                           --results results/vivit_b16_summary.json \
                           --out checkpoints/release/vivit_b16_ucf101_fp16.pth

The output loads like any other checkpoint (``utils.load_model_from_checkpoint``). The weights are
cast back to fp32 on load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--results", default=None, help="evaluate.py summary JSON; its test metrics are embedded")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metrics = dict(ckpt.get("metrics", {}))
    if args.results:
        with open(args.results) as f:
            s = json.load(f)
        metrics.update(test_top1=s["top1"], test_top5=s["top5"], test_split=f"{s['split']}/{s.get('split_mode')}")

    slim = {
        "model_name": ckpt["model_name"],
        "model_cfg": ckpt["model_cfg"],
        "classes": ckpt["classes"],
        "model_state_dict": {k: v.half() if v.is_floating_point() else v for k, v in ckpt["model_state_dict"].items()},
        "epoch": ckpt.get("epoch"),
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(slim, args.out)

    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"Wrote {args.out}  ({os.path.getsize(args.out) / 2**20:.1f} MiB)  sha256={sha}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
