"""Web demo: upload a video, get the top-5 UCF101 actions.

    python app.py --checkpoint checkpoints/<run>/best.pth            # http://127.0.0.1:7860
    python app.py --checkpoint ... --host 0.0.0.0 --port 7860        # serve on the LAN / in Docker

Gradio also exposes a REST endpoint: see "Use via API" at the bottom of the page.
"""

from __future__ import annotations

import argparse
import glob
import os

import gradio as gr

from inference import ActionRecognizer


def latest_checkpoint(root="checkpoints"):
    cands = glob.glob(os.path.join(root, "*", "best.pth"))
    return max(cands, key=os.path.getmtime) if cands else None


def build_ui(rec: ActionRecognizer, examples: list[str]):
    def classify(video_path, num_clips):
        if not video_path:
            return {}
        preds = rec.predict(video_path, top_k=5, num_clips=int(num_clips))
        return {name: prob for name, prob in preds}

    acc = rec.metrics.get("val_top1")
    subtitle = f"Model: `{rec.model_name}` on {rec.device}" + (f" | val top-1 {acc:.1f}%" if acc else "")
    with gr.Blocks(title="UCF101 Action Recognition") as demo:
        gr.Markdown(f"# Video Vision Transformer: Action Recognition\n{subtitle}\n\n"
                    f"Upload a short clip of a human action (one of {len(rec.classes)} UCF101 classes).")
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Input video", sources=["upload"])
                clips = gr.Slider(1, 5, value=rec.num_clips, step=1, label="Temporal views (more = slower, steadier)")
                btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                out = gr.Label(num_top_classes=5, label="Top-5 predictions")
        btn.click(classify, [video, clips], out, api_name="classify")
        if examples:
            gr.Examples(examples=[[e, rec.num_clips] for e in examples], inputs=[video, clips], outputs=out,
                        fn=classify, cache_examples=False)
        with gr.Accordion("Classes", open=False):
            gr.Markdown(", ".join(rec.classes))
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT") or latest_checkpoint())
    p.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    p.add_argument("--num_clips", type=int, default=3)
    p.add_argument("--examples_dir", default="./datasets/ucf101/test")
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = p.parse_args()
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise SystemExit("No checkpoint found. Train first (python train.py) or pass --checkpoint.")

    rec = ActionRecognizer(args.checkpoint, args.device, args.num_clips)
    # Only clips from the held-out groups g01-g07 (never seen in training under the official split).
    examples = sorted(glob.glob(os.path.join(args.examples_dir, "*", "*_g0[1-7]_*.avi")))[::300][:6]
    build_ui(rec, examples).queue().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
