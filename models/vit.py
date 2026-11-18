"""Vision Transformer building blocks and a video ViT (ViViT factorized encoder).

The image ViT follows the ViT_Flowers102 notebook: Conv2d patch embedding plus a
class token and learnable position embedding, then pre-norm encoder blocks
(multi-head self-attention, then MLP, each with a residual connection), then a
LayerNorm + Linear head on the class token.

Changes from the notebook:
  * Attention goes through ``F.scaled_dot_product_attention`` (fused kernels,
    much less memory), with stochastic depth and optional gradient checkpointing.
  * ``load_torchvision_vit_weights`` copies ImageNet ViT-B/16 weights from
    torchvision into these modules (the notebook's transfer-learning step).
  * ``VideoVisionTransformer`` extends the ViT to video, following the
    "factorized encoder" design from ViViT (Arnab et al., 2021). A spatial ViT
    turns each frame into a class token, and a temporal transformer then
    attends across those per-frame tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class DropPath(nn.Module):
    """Stochastic depth: randomly drop the residual branch per sample."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class PatchEmbedding(nn.Module):
    """Image -> patch tokens, with a class token and position embedding prepended or added."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        channels: int = 3,
        embedding_size: int = 768,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patcher = nn.Conv2d(channels, embedding_size, patch_size, patch_size, 0)
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, embedding_size))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embedding_size))
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % self.patch_size == 0, (
            f"Image size must be divisible by patch size. Got {tuple(x.shape)}, patch {self.patch_size}"
        )
        x = self.patcher(x).flatten(2).transpose(1, 2)  # [B, N, D]
        cls = self.class_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + _resize_pos_embed(self.position_embedding, x.shape[1])
        return self.embedding_dropout(x)


class MultiHeadSelfAttentionBlock(nn.Module):
    """Pre-norm MSA block: ``x + DropPath(MSA(LN(x)))``."""

    def __init__(self, embedding_size=768, num_heads=12, attn_dropout=0.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        assert embedding_size % num_heads == 0
        self.num_heads = num_heads
        self.layernorm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.qkv = nn.Linear(embedding_size, embedding_size * 3)
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.attn_dropout = attn_dropout
        self.proj_dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.layernorm(x)).reshape(b, n, 3, self.num_heads, d // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, H, N, Dh]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(b, n, d)
        return x + self.drop_path(self.proj_dropout(self.proj(out)))


class MultiLayerPerceptronBlock(nn.Module):
    """Pre-norm MLP block: ``x + DropPath(MLP(LN(x)))``."""

    def __init__(self, embedding_size=768, hidden_units=3072, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.layernorm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, embedding_size),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.mlp(self.layernorm(x)))


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size=768, num_heads=12, hidden_units=3072, dropout=0.0, attn_dropout=0.0, drop_path=0.0):
        super().__init__()
        self.msa_block = MultiHeadSelfAttentionBlock(embedding_size, num_heads, attn_dropout, dropout, drop_path)
        self.mlp_block = MultiLayerPerceptronBlock(embedding_size, hidden_units, dropout, drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_block(self.msa_block(x))


class EncoderStack(nn.Module):
    def __init__(self, depth, embedding_size, num_heads, hidden_units, dropout, attn_dropout, drop_path_rates):
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerEncoder(embedding_size, num_heads, hidden_units, dropout, attn_dropout, drop_path_rates[i])
            for i in range(depth)
        )
        self.norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.grad_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.grad_checkpointing and self.training and x.requires_grad:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.norm(x)


class VisionTransformer(nn.Module):
    """Image ViT from the notebook (ViT-B/16 by default)."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        channels=3,
        embedding_size=768,
        num_heads=12,
        hidden_units=3072,
        num_encoders=12,
        num_classes=1000,
        dropout=0.0,
        attn_dropout=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.patch_embedding_layer = PatchEmbedding(img_size, patch_size, channels, embedding_size, dropout)
        dpr = torch.linspace(0, drop_path, num_encoders).tolist()
        self.transformer_encoder = EncoderStack(
            num_encoders, embedding_size, num_heads, hidden_units, dropout, attn_dropout, dpr
        )
        self.classifier = nn.Linear(embedding_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (normalized) class token, shape ``[B, D]``."""
        x = self.patch_embedding_layer(x)
        x = self.transformer_encoder(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class VideoVisionTransformer(nn.Module):
    """ViViT factorized encoder.

    ``[B, C, T, H, W]`` -> spatial ViT on every frame -> ``T`` frame tokens
    -> temporal class token plus temporal position embedding -> temporal
    transformer -> classification head.
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int = 8,
        img_size: int = 224,
        patch_size: int = 16,
        embedding_size: int = 768,
        num_heads: int = 12,
        hidden_units: int = 3072,
        spatial_depth: int = 12,
        temporal_depth: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.1,
        head_dropout: float = 0.5,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.spatial = VisionTransformer(
            img_size, patch_size, 3, embedding_size, num_heads, hidden_units, spatial_depth,
            num_classes=0, dropout=dropout, attn_dropout=attn_dropout, drop_path=drop_path,
        )
        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, embedding_size))
        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames + 1, embedding_size))
        dpr = torch.linspace(0, drop_path, temporal_depth).tolist()
        self.temporal = EncoderStack(
            temporal_depth, embedding_size, num_heads, hidden_units, dropout, attn_dropout, dpr
        )
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(embedding_size, num_classes))

        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_cls, std=0.02)
        self.temporal.apply(_init_linear)
        self.head.apply(_init_linear)

    def set_grad_checkpointing(self, enabled: bool = True):
        self.spatial.transformer_encoder.grad_checkpointing = enabled
        self.temporal.grad_checkpointing = enabled

    def head_parameters(self):
        """Newly initialized parameters, which train with a higher learning rate."""
        yield self.temporal_cls
        yield self.temporal_pos
        yield from self.temporal.parameters()
        yield from self.head.parameters()

    def freeze_backbone(self, freeze: bool = True):
        for p in self.spatial.parameters():
            p.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        frames = x.transpose(1, 2).reshape(b * t, c, h, w)
        tokens = self.spatial.forward_features(frames).view(b, t, -1)  # [B, T, D]
        cls = self.temporal_cls.expand(b, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + _resize_pos_embed(self.temporal_pos, tokens.shape[1], has_grid=False)
        tokens = self.temporal(tokens)
        return self.head(tokens[:, 0])


def _init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def _resize_pos_embed(pos: torch.Tensor, length: int, has_grid: bool = True) -> torch.Tensor:
    """Interpolate a ``[1, 1 + N, D]`` position embedding to ``1 + M`` tokens (keeping the class slot)."""
    if pos.shape[1] == length:
        return pos
    cls_pos, grid = pos[:, :1], pos[:, 1:]
    n_old, n_new, d = grid.shape[1], length - 1, grid.shape[2]
    if has_grid:
        s_old, s_new = int(math.sqrt(n_old)), int(math.sqrt(n_new))
        grid = grid.reshape(1, s_old, s_old, d).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(s_new, s_new), mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, n_new, d)
    else:
        grid = F.interpolate(grid.transpose(1, 2), size=n_new, mode="linear", align_corners=False).transpose(1, 2)
    return torch.cat((cls_pos, grid), dim=1)


@torch.no_grad()
def load_torchvision_vit_weights(vit: VisionTransformer, weights="DEFAULT") -> VisionTransformer:
    """Copy torchvision ``vit_b_16`` ImageNet weights into our ``VisionTransformer``."""
    import torchvision

    w = torchvision.models.ViT_B_16_Weights.DEFAULT if weights == "DEFAULT" else weights
    src = torchvision.models.vit_b_16(weights=w).state_dict()

    pe = vit.patch_embedding_layer
    pe.patcher.weight.copy_(src["conv_proj.weight"])
    pe.patcher.bias.copy_(src["conv_proj.bias"])
    pe.class_embedding.copy_(src["class_token"])
    pe.position_embedding.copy_(_resize_pos_embed(src["encoder.pos_embedding"], pe.position_embedding.shape[1]))

    enc = vit.transformer_encoder
    for i, layer in enumerate(enc.layers):
        p = f"encoder.layers.encoder_layer_{i}."
        msa, mlp = layer.msa_block, layer.mlp_block
        msa.layernorm.weight.copy_(src[p + "ln_1.weight"])
        msa.layernorm.bias.copy_(src[p + "ln_1.bias"])
        msa.qkv.weight.copy_(src[p + "self_attention.in_proj_weight"])
        msa.qkv.bias.copy_(src[p + "self_attention.in_proj_bias"])
        msa.proj.weight.copy_(src[p + "self_attention.out_proj.weight"])
        msa.proj.bias.copy_(src[p + "self_attention.out_proj.bias"])
        mlp.layernorm.weight.copy_(src[p + "ln_2.weight"])
        mlp.layernorm.bias.copy_(src[p + "ln_2.bias"])
        mlp.mlp[0].weight.copy_(src[p + "mlp.0.weight"])
        mlp.mlp[0].bias.copy_(src[p + "mlp.0.bias"])
        mlp.mlp[3].weight.copy_(src[p + "mlp.3.weight"])
        mlp.mlp[3].bias.copy_(src[p + "mlp.3.bias"])
    enc.norm.weight.copy_(src["encoder.ln.weight"])
    enc.norm.bias.copy_(src["encoder.ln.bias"])
    return vit
