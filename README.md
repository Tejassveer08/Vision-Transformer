# Video Vision Transformer: UCF101 Action Recognition

This project classifies human actions in video clips into the 101 UCF101 classes
([Kaggle: matthewjansen/ucf101-action-recognition](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition)).
It covers the full workflow: data download, training, multi-view evaluation, CLI inference, and a web app.

The main model, **`vivit_b16`**, extends the ViT built in `ViT_Flowers102.ipynb` from images to video:

```
clip [B, 3, T=8, 224, 224]
  └─ every frame → ViT-B/16 (patch embedding → 12 × [MSA → MLP]) → frame CLS token   ← ImageNet weights (torchvision)
       └─ [temporal CLS, f1 … f8] + temporal position embedding
            └─ 4 × temporal transformer blocks → LayerNorm → Dropout → Linear(101)
```

This is the *factorized encoder* design from ViViT (Arnab et al., 2021). The spatial encoder uses the notebook's modules
(`PatchEmbedding`, `MultiHeadSelfAttentionBlock`, `MultiLayerPerceptronBlock`), extended with fused SDPA attention,
stochastic depth and gradient checkpointing. The notebook's transfer-learning step is also built in:
`load_torchvision_vit_weights` copies ImageNet ViT-B/16 weights into these modules, and the copied model matches
torchvision's outputs to about 1e-6.

## Results (`vivit_b16`, group-disjoint split)

Trained for 6 epochs (about 2.2 h on an RTX 3050 Laptop 6 GB) with `python train.py --model vivit_b16 --epochs 6`.

| Epoch | Train loss | Train acc | Val top-1 | Val top-5 |
|------:|-----------:|----------:|----------:|----------:|
| 1 | 2.01 | 65.4% | 74.3% | 94.4% |
| 2 | 1.27 | 86.9% | 76.0% | 93.2% |
| 3 | 1.06 | 93.9% | 81.2% | 94.3% |
| 4 | 0.94 | 97.3% | 81.4% | 95.7% |
| 5 | 0.86 | 98.7% | 83.4% | 96.7% |
| 6 | 0.84 | 99.0% | **83.8%** | 97.1% |

**Test (UCF101 split-1 test groups g01–g07, 3,820 videos, 3 temporal views): top-1 86.4%, top-5 96.4%.**

The most common errors are between look-alike classes: BasketballDunk → Basketball (36/37), CricketShot ↔ CricketBowling,
Shotput → ThrowDiscus, BrushingTeeth → ShavingBeard, BlowDryHair → Haircut. The full report and confusion matrix are in
`results/`. The model reaches 99% train accuracy, so it overfits. More augmentation (mixup or RandAugment), more frames,
or a Kinetics-pretrained backbone (`mvit_v2_s`) are the obvious next steps.

### Other models in the zoo (`models/video_models.py`)

| `--model`      | Architecture                         | Pretraining  | Clip         | Train step* |
|----------------|--------------------------------------|--------------|--------------|-------------|
| `vivit_b16`    | ViViT factorized encoder (ViT-B/16)  | ImageNet-1k  | 8 seg, 224px | 0.39 s, 3.6 GB |
| `vivit_tiny`   | small ViViT, from scratch            | none         | 8 seg, 112px | 0.06 s |
| `mvit_v2_s`    | Multiscale ViT v2 (torchvision)      | Kinetics-400 | 16×4, 224px  | 1.2 s, 5.0 GB |
| `x3d_m`        | X3D-M ([PyTorchVideo hub](https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/)) | Kinetics-400 | 16×5, 224px | 0.23 s, 1.7 GB |
| `slowfast_r50` | SlowFast R50 ([PyTorchVideo hub](https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/)) | Kinetics-400 | 32×2, 224px | 0.49 s, 2.7 GB |

\*batch 4, bf16, RTX 3050 Laptop 6 GB.

## Important: the split

The Kaggle folders split **randomly per clip**. UCF101 clips come in 25 *groups* per class, and clips within a group
share the same actor and background. In the Kaggle split, 1,272 of the 1,273 test groups also appear in train, so
test accuracy on it is inflated. By default (`--split_mode official`) the code pools all Kaggle videos and re-splits
them **by group**, following the official UCF101 split 1:

| split | groups  | videos |
|-------|---------|--------|
| train | g11–g25 | 7,991  |
| val   | g08–g10 | 1,640  |
| test  | g01–g07 | 3,820 (official split-1 test set) |

To use the Kaggle folders as they are, pass `--split_mode kaggle`.

## Setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

To get the dataset (about 7 GB, needs `~/.kaggle/kaggle.json`):

```bash
python -m data.ucf101 --download        # → datasets/ucf101/{train,val,test}/<Class>/*.avi
```

## Train

```bash
python train.py --model vivit_b16 --epochs 6 --batch_size 4 --accum_steps 4 --run_name vivit_b16_ucf101
```

- AdamW with cosine LR and warmup. New layers (temporal transformer and head) train at 10× the backbone LR.
- Uses bf16 autocast, label smoothing 0.1, gradient clipping, scale-jitter crop, flip and colour jitter.
- If memory is tight, add `--grad_ckpt` (2.2 GB peak instead of 3.6 GB for ViViT) or `--freeze_backbone`.
- Resume with `--resume checkpoints/<run>/last.pth`. Watch training with `tensorboard --logdir logs`.
- Each checkpoint stores the model name, preprocessing config and class names, so inference needs no other files.

## Evaluate (test split, 3 temporal views per video)

```bash
python evaluate.py --checkpoint checkpoints/vivit_b16_ucf101/best.pth --num_clips 3
```

This writes `results/<model>_classification_report.txt`, `_confusion_matrix.png`, `_summary.json` (top-1, top-5, and the
worst classes).

## Inference and deployment

```bash
python inference.py --checkpoint checkpoints/vivit_b16_ucf101/best.pth --video my_clip.mp4
python app.py --checkpoint checkpoints/vivit_b16_ucf101/best.pth    # Gradio UI at http://127.0.0.1:7860
```

The Gradio app also serves an API endpoint (`/classify`; see "Use via API" on the page):

```python
from gradio_client import Client, handle_file
Client("http://127.0.0.1:7860/").predict(handle_file("clip.avi"), 3, api_name="/classify")
```

To use the model in-process, without the server:

```python
from inference import ActionRecognizer
rec = ActionRecognizer("checkpoints/vivit_b16_ucf101/best.pth")
rec.predict("clip.avi")   # [('Basketball', 0.93), ...]
```

To run the app in Docker (the image includes `checkpoints/*/best.pth`):

```bash
docker build -t ucf101-vit .
docker run --gpus all -p 7860:7860 ucf101-vit
```

## Layout

```
data/transforms.py     custom video transforms (PyTorch custom-transform recipe style)
data/ucf101.py         dataset, frame sampling (segment / dense), group-based splits, Kaggle download
models/vit.py          ViT blocks from the notebook, ViViT, ImageNet weight loading
models/video_models.py model zoo and per-model preprocessing configs
train.py / evaluate.py / inference.py / app.py
```
