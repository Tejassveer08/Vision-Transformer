"""Model zoo for UCF101 action recognition.

| name            | family                     | pretraining  | input            |
|-----------------|----------------------------|--------------|------------------|
| vivit_b16       | ViT-B/16 + temporal ViT    | ImageNet-1k  | 8 frames, 224px  |
| vivit_tiny      | small ViViT, from scratch  | none         | 8 frames, 112px  |
| mvit_v2_s       | Multiscale ViT v2 (video)  | Kinetics-400 | 16x4, 224px      |
| x3d_m           | X3D-M (PyTorchVideo hub)   | Kinetics-400 | 16x5, 224px      |
| slowfast_r50    | SlowFast R50 (PTV hub)     | Kinetics-400 | 32x2, 224px      |

Each model comes with a ``VideoModelConfig`` that the data pipeline and the
inference code read, so a checkpoint carries everything needed to preprocess
its inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from models.vit import VideoVisionTransformer, load_torchvision_vit_weights

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
KINETICS_MEAN, KINETICS_STD = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)


@dataclass
class VideoModelConfig:
    num_frames: int = 8
    # "segment": split the video into num_frames equal segments and take one frame from each
    #            (random within the segment when training, the middle frame when evaluating).
    # "dense":   take num_frames consecutive frames, sampling_rate apart.
    sampling: str = "segment"
    sampling_rate: int = 1
    crop_size: int = 224
    resize_size: int = 256
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["mean"], d["std"] = tuple(d["mean"]), tuple(d["std"])
        return cls(**d)


MODEL_CONFIGS = {
    "vivit_b16": VideoModelConfig(num_frames=8, sampling="segment"),
    "vivit_tiny": VideoModelConfig(num_frames=8, sampling="segment", crop_size=112, resize_size=128),
    "mvit_v2_s": VideoModelConfig(num_frames=16, sampling="dense", sampling_rate=4, mean=KINETICS_MEAN, std=KINETICS_STD),
    "x3d_m": VideoModelConfig(num_frames=16, sampling="dense", sampling_rate=5, mean=KINETICS_MEAN, std=KINETICS_STD),
    "slowfast_r50": VideoModelConfig(
        num_frames=32, sampling="dense", sampling_rate=2, mean=KINETICS_MEAN, std=KINETICS_STD, extra={"alpha": 4}
    ),
}


class HubVideoModel(nn.Module):
    """Wraps torchvision / PyTorchVideo backbones behind one interface."""

    def __init__(self, name: str, num_classes: int, pretrained: bool, cfg: VideoModelConfig):
        super().__init__()
        self.name = name
        self.cfg = cfg
        if name == "mvit_v2_s":
            from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

            self.model = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT if pretrained else None)
            in_features = self.model.head[-1].in_features
            self.model.head[-1] = nn.Linear(in_features, num_classes)
            self._head = self.model.head
        elif name in ("x3d_m", "slowfast_r50"):
            self.model = torch.hub.load("facebookresearch/pytorchvideo", name, pretrained=pretrained, trust_repo=True)
            head = self.model.blocks[-1]
            head.proj = nn.Linear(head.proj.in_features, num_classes)
            self._head = head.proj  # only the new layer; X3D's projected-pool convs stay at backbone LR
        else:
            raise ValueError(f"Unknown hub model {name}")

    def head_parameters(self):
        return self._head.parameters()

    def freeze_backbone(self, freeze: bool = True):
        for p in self.model.parameters():
            p.requires_grad = not freeze
        for p in self._head.parameters():
            p.requires_grad = True

    def set_grad_checkpointing(self, enabled: bool = True):
        pass  # not supported for these backbones

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "slowfast_r50":
            # PackPathway: the fast pathway sees every frame, the slow pathway every alpha-th frame.
            alpha = self.cfg.extra.get("alpha", 4)
            t = x.shape[2]
            idx = torch.linspace(0, t - 1, t // alpha, device=x.device).long()
            return self.model([x.index_select(2, idx), x])
        return self.model(x)


def get_model(model_name: str, num_classes: int = 101, pretrained: bool = True, cfg: VideoModelConfig | None = None):
    """Build a model and attach its preprocessing config as ``model.cfg``."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {list(MODEL_CONFIGS)}")
    cfg = cfg or VideoModelConfig.from_dict(MODEL_CONFIGS[model_name].to_dict())

    if model_name == "vivit_b16":
        model = VideoVisionTransformer(num_classes=num_classes, num_frames=cfg.num_frames, img_size=cfg.crop_size)
        if pretrained:
            load_torchvision_vit_weights(model.spatial)
    elif model_name == "vivit_tiny":
        model = VideoVisionTransformer(
            num_classes=num_classes, num_frames=cfg.num_frames, img_size=cfg.crop_size, patch_size=16,
            embedding_size=192, num_heads=3, hidden_units=768, spatial_depth=6, temporal_depth=2,
            drop_path=0.1, head_dropout=0.2,
        )
    else:
        model = HubVideoModel(model_name, num_classes, pretrained, cfg)

    model.cfg = cfg
    model.model_name = model_name
    return model
