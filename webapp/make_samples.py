"""Bundle a few held-out UCF101 test clips with the web app, as browser-playable H.264 MP4.

    python webapp/make_samples.py            # writes webapp/samples/*.mp4

The clips come from groups g01-g07 (the official split-1 test groups), so the model never trained on them.
UCF101 credit: Soomro, Zamir & Shah, "UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild", 2012.
"""

from __future__ import annotations

import argparse
import glob
from fractions import Fraction
from pathlib import Path

import av

DEFAULT_CLASSES = [
    "PlayingGuitar", "FrontCrawl", "ApplyEyeMakeup", "HorseRiding", "SkyDiving", "Drumming",
    "Billiards", "JumpingJack", "PizzaTossing", "TaiChi", "Archery", "Basketball",
]


def to_mp4(src: str, dst: Path, crf: int = 28):
    with av.open(src) as inp, av.open(str(dst), "w") as out:
        vin = inp.streams.video[0]
        rate = vin.average_rate or 25
        s = out.add_stream("libx264", rate=int(round(float(rate))))
        s.width, s.height, s.pix_fmt = vin.codec_context.width, vin.codec_context.height, "yuv420p"
        s.options = {"crf": str(crf), "preset": "slow", "movflags": "+faststart"}
        for i, frame in enumerate(inp.decode(video=0)):
            frame = frame.reformat(format="yuv420p")
            frame.pts, frame.time_base = i, Fraction(1, int(round(float(rate))))  # source AVI timestamps can be broken
            for pkt in s.encode(frame):
                out.mux(pkt)
        for pkt in s.encode():
            out.mux(pkt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./datasets/ucf101")
    p.add_argument("--out", default="./webapp/samples")
    p.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for cls in args.classes:
        # Any Kaggle folder can hold held-out groups; take the first g01-g07 clip of this class.
        cands = sorted(glob.glob(f"{args.data_root}/*/{cls}/v_{cls}_g0[1-7]_c*.avi"), key=lambda p: Path(p).name)
        if not cands:
            print(f"[skip] no held-out clip for {cls}")
            continue
        src = cands[len(cands) // 2]
        dst = out / (Path(src).stem + ".mp4")
        to_mp4(src, dst)
        print(f"{dst.name}: {dst.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
