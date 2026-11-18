"""Streamlit front end for the UCF101 video vision transformer.

Run locally (from the repo root):
    streamlit run webapp/streamlit_app.py

Where the model comes from, checked in this order:
    1. MODEL_PATH (env var or Streamlit secret): a local checkpoint file
    2. the newest checkpoints/*/best*.pth in this repo (local development)
    3. MODEL_URL  (env var or Streamlit secret): downloaded once and cached. Streamlit Cloud uses this.
"""

from __future__ import annotations

import glob
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `data`, `models`, `inference` import when run from anywhere

from data.ucf101 import load_clip  # noqa: E402
from inference import ActionRecognizer  # noqa: E402

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", Path.home() / ".cache" / "ucf101-vit"))
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]
BROWSER_PLAYABLE = {"mp4", "webm", "mov"}

st.set_page_config(page_title="Video ViT · Action Recognition", page_icon="🎬", layout="wide")


def setting(name: str, default=None):
    """Read a setting from env vars first, then Streamlit secrets."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, default)
    except Exception:  # no secrets.toml
        return default


def resolve_checkpoint() -> str:
    path = setting("MODEL_PATH")
    if path and Path(path).exists():
        return path
    local = glob.glob(str(ROOT / "checkpoints" / "*" / "*.pth"))
    local = [p for p in local if Path(p).name != "last.pth"]  # last.pth includes optimizer state; skip it
    if local:
        return max(local, key=os.path.getmtime)
    url = setting("MODEL_URL")
    if url:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dest = CACHE_DIR / Path(url.split("?")[0]).name
        if not dest.exists():
            import urllib.error

            import torch

            with st.spinner(f"Downloading model weights (first start only) from {url} ..."):
                tmp = dest.with_suffix(".part")
                try:
                    torch.hub.download_url_to_file(url, str(tmp), progress=False)
                except urllib.error.HTTPError as e:
                    hint = (" The file isn't at that address. Check the release tag and file name in MODEL_URL, "
                            "and that the release is published and the repo is public.") if e.code == 404 else ""
                    st.error(f"Could not download the model weights (HTTP {e.code}) from `{url}`.{hint}")
                    st.stop()
                except (urllib.error.URLError, OSError) as e:
                    st.error(f"Could not download the model weights from `{url}`: {e}")
                    st.stop()
                tmp.replace(dest)
        return str(dest)
    st.error("No model found. Set MODEL_URL (or MODEL_PATH) in the environment or in Streamlit secrets, "
             "or train one locally with `python train.py`.")
    st.stop()


@st.cache_resource(show_spinner="Loading the vision transformer ...")
def get_recognizer(path: str) -> ActionRecognizer:
    import torch

    torch.set_num_threads(max(os.cpu_count() or 1, 1))
    return ActionRecognizer(path, device=setting("DEVICE", "cuda"), num_clips=1)


def frame_strip(video_path: str, rec: ActionRecognizer):
    """The exact frames the model looks at (view 0), for display."""
    return list(load_clip(video_path, rec.cfg, train=False, clip_idx=0, num_clips=1))


def sample_clips():
    """Held-out test clips (groups g01-g07): the MP4s bundled in webapp/samples, plus more from the dataset if present."""
    bundled = sorted(glob.glob(str(SAMPLES_DIR / "*.mp4")))
    local = sorted(glob.glob(str(ROOT / "datasets" / "ucf101" / "test" / "*" / "*_g0[1-7]_*.avi")))[::150][:25]
    return bundled + local


def true_label(path: str) -> str:
    m = re.match(r"v_(.+?)_g\d+_c\d+", Path(path).stem)
    return m.group(1) if m else Path(path).parent.name


# ----------------------------------------------------------------------------- UI
rec = get_recognizer(resolve_checkpoint())
metrics = rec.metrics or {}

st.title("🎬 Video Vision Transformer: Action Recognition")
st.caption(
    "A ViViT model (ViT-B/16 per frame plus a temporal transformer) fine-tuned on UCF101. "
    "Upload a short clip of a person doing something, and it predicts which of 101 actions it is."
)

with st.sidebar:
    st.header("Model")
    st.markdown(f"**{rec.model_name}** on `{rec.device}`")
    if metrics.get("test_top1"):
        c1, c2 = st.columns(2)
        c1.metric("Test top-1", f"{metrics['test_top1']:.1f}%")
        c2.metric("Test top-5", f"{metrics['test_top5']:.1f}%")
        st.caption("Test set: UCF101 split-1 held-out groups (3,820 clips), 3 views per clip.")
    elif metrics.get("val_top1"):
        st.metric("Validation top-1", f"{metrics['val_top1']:.1f}%")
    st.divider()
    default_views = 1 if rec.device.type == "cpu" else 3
    views = st.slider("Temporal views", 1, 5, default_views,
                      help="How many evenly spaced 8-frame windows to average. More views are steadier but slower.")
    top_k = st.slider("Show top-k", 3, 10, 5)
    with st.expander(f"All {len(rec.classes)} classes"):
        st.write(", ".join(rec.classes))

samples = sample_clips()
source = "Upload a video"
if samples:
    source = st.radio("Video source", ["Upload a video", "Sample from the test set"], horizontal=True,
                      label_visibility="collapsed")

video_path, display_name, ext = None, None, None
if source == "Upload a video":
    up = st.file_uploader("Upload a video", type=VIDEO_TYPES)
    if up is not None:
        ext = up.name.rsplit(".", 1)[-1].lower()
        safe_id = "".join(c for c in str(up.file_id) if c.isalnum())
        tmp = Path(tempfile.gettempdir()) / f"st_upload_{safe_id}.{ext}"
        if not tmp.exists():
            tmp.write_bytes(up.getbuffer())
        video_path, display_name = str(tmp), up.name
else:
    st.caption("Clips from UCF101 test groups the model never saw in training "
               "(Soomro, Zamir & Shah, 2012).")
    by_label = {f"{true_label(p)}  ·  {Path(p).name}": p for p in samples}
    choice = by_label[st.selectbox("Held-out clip", list(by_label))]
    video_path, display_name, ext = choice, Path(choice).name, Path(choice).suffix.lstrip(".").lower()

if video_path is None:
    st.info("Upload a clip to get started. Works best on 2 to 10 second clips of a single action, "
            "for example sports, playing an instrument, or exercise.")
    st.stop()

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader(display_name)
    if ext in BROWSER_PLAYABLE:
        st.video(video_path)
    try:
        frames = frame_strip(video_path, rec)
    except Exception as e:  # unreadable / corrupt file
        st.error(f"Could not read this video: {e}")
        st.stop()
    st.caption("The frames the model sees (8 evenly spaced across the clip):")
    cols = st.columns(4)
    for i, f in enumerate(frames):
        cols[i % 4].image(f, width="stretch")

with right:
    st.subheader("Prediction")
    t0 = time.time()
    with st.spinner("Running the transformer ..."):
        preds = rec.predict(video_path, top_k=top_k, num_clips=views)
    dt = time.time() - t0
    best, p = preds[0]
    st.markdown(f"### {best}")
    st.caption(f"{p * 100:.1f}% confidence · {views} view(s) · {dt:.1f}s on {rec.device.type}")
    for name, prob in preds:
        st.progress(prob, text=f"{name}: {prob * 100:.1f}%")
    if source != "Upload a video":
        truth = true_label(video_path)
        (st.success if truth == best else st.warning)(f"Ground truth: **{truth}**")
