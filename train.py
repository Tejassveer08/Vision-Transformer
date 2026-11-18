"""Train an action-recognition model on UCF101.

Example (default: ViViT with ImageNet ViT-B/16 spatial weights, sized for a 6 GB GPU)::

    python train.py --model vivit_b16 --epochs 6 --batch_size 4 --grad_ckpt

Checkpoints go to ``checkpoints/<run>/{best,last}.pth``; TensorBoard logs go to ``logs/<run>``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.ucf101 import get_dataloaders
from models.video_models import MODEL_CONFIGS, VideoModelConfig, get_model
from utils import autocast_dtype, get_device, save_checkpoint, set_seed, topk_correct


def build_optimizer(model, args):
    head_ids = {id(p) for p in model.head_parameters()}
    groups = {k: [] for k in ("bb_decay", "bb_no_decay", "head_decay", "head_no_decay")}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = id(p) in head_ids
        no_decay = p.ndim <= 1 or "position" in name or "pos" in name.split(".")[-1] or "cls" in name or "class_embedding" in name
        key = ("head" if is_head else "bb") + ("_no_decay" if no_decay else "_decay")
        groups[key].append(p)
    lr_bb, lr_head = args.lr, args.lr * args.head_lr_mult
    param_groups = [
        {"params": groups["bb_decay"], "lr": lr_bb, "weight_decay": args.weight_decay},
        {"params": groups["bb_no_decay"], "lr": lr_bb, "weight_decay": 0.0},
        {"params": groups["head_decay"], "lr": lr_head, "weight_decay": args.weight_decay},
        {"params": groups["head_no_decay"], "lr": lr_head, "weight_decay": 0.0},
    ]
    param_groups = [g for g in param_groups if g["params"]]
    for g in param_groups:
        g["base_lr"] = g["lr"]
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp_dtype, epoch, args, state, writer):
    model.train()
    total_loss = correct = seen = 0
    max_steps = args.max_train_steps or len(loader)
    pbar = tqdm(loader, total=max_steps, desc=f"Epoch {epoch} [train]", dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    for i, (clips, targets) in enumerate(pbar):
        if i >= max_steps:
            break
        clips = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
        with ctx:
            logits = model(clips)
            loss = criterion(logits.float(), targets)
        scaler.scale(loss / args.accum_steps).backward()

        if (i + 1) % args.accum_steps == 0 or i + 1 == max_steps:
            f = lr_factor(state["step"], state["total_steps"], state["warmup_steps"])
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * f
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            state["step"] += 1

        bs = targets.size(0)
        total_loss += loss.item() * bs
        correct += topk_correct(logits, targets, (1,))[0]
        seen += bs
        if writer and i % 20 == 0:
            gstep = (epoch - 1) * max_steps + i
            writer.add_scalar("train/loss_step", loss.item(), gstep)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], gstep)
        pbar.set_postfix(loss=f"{total_loss / seen:.3f}", acc=f"{100 * correct / seen:.1f}%",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    return total_loss / max(seen, 1), 100.0 * correct / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype, desc="val", max_batches=None):
    """Return ``(loss, top1, top5)``. Accepts batches of ``[B, C, T, H, W]`` or multi-view ``[B, V, C, T, H, W]``."""
    model.eval()
    total_loss = c1 = c5 = seen = 0
    n = min(max_batches or len(loader), len(loader))
    for i, (clips, targets) in enumerate(tqdm(loader, total=n, desc=f"[{desc}]", dynamic_ncols=True)):
        if i >= n:
            break
        clips = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
        with ctx:
            if clips.ndim == 6:
                b, v = clips.shape[:2]
                logits = model(clips.flatten(0, 1)).float().view(b, v, -1)
                probs = logits.softmax(-1).mean(1)
                loss = nn.functional.nll_loss(probs.clamp_min(1e-8).log(), targets) * b
                logits = probs
            else:
                logits = model(clips).float()
                loss = criterion(logits, targets) * targets.size(0)
        total_loss += loss.item()
        k1, k5 = topk_correct(logits, targets, (1, 5))
        c1, c5, seen = c1 + k1, c5 + k5, seen + targets.size(0)
    return total_loss / max(seen, 1), 100.0 * c1 / max(seen, 1), 100.0 * c5 / max(seen, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Train video models on UCF101")
    p.add_argument("--data_root", default="./datasets/ucf101")
    p.add_argument("--split_mode", default="official", choices=["official", "kaggle"],
                   help="official: group-disjoint split (test=g01-07, val=g08-10, train=g11-25); "
                        "kaggle: the dataset's own random per-clip folders (group leakage)")
    p.add_argument("--model", default="vivit_b16", choices=list(MODEL_CONFIGS))
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_frames", type=int, default=None, help="Override the model's default clip length")
    p.add_argument("--freeze_backbone", action="store_true", help="Train only the new (temporal/head) layers")
    p.add_argument("--grad_ckpt", action="store_true", help="Gradient checkpointing (ViViT) to save memory")

    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation (effective batch = batch_size*accum)")
    p.add_argument("--lr", type=float, default=5e-5, help="Backbone learning rate")
    p.add_argument("--head_lr_mult", type=float, default=10.0, help="LR multiplier for newly initialized layers")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=float, default=0.5)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true")

    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", default=None)
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--log_dir", default="./logs")
    p.add_argument("--resume", default=None, help="Path to a last.pth to resume from")
    p.add_argument("--max_train_steps", type=int, default=None, help="Limit batches per epoch (smoke tests)")
    p.add_argument("--max_val_batches", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    amp_dtype = None if args.no_amp else autocast_dtype(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = VideoModelConfig.from_dict(MODEL_CONFIGS[args.model].to_dict())
    if args.num_frames:
        cfg.num_frames = args.num_frames

    run = args.run_name or f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = os.path.join(args.checkpoint_dir, run)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Run: {run} | device: {device} | amp: {amp_dtype}")

    loaders, classes = get_dataloaders(args.data_root, cfg, args.batch_size, args.num_workers,
                                       eval_batch_size=args.eval_batch_size, splits=("train", "val"),
                                       split_mode=args.split_mode)
    print(f"{len(classes)} classes | train {len(loaders['train'].dataset)} | val {len(loaders['val'].dataset)}")

    model = get_model(args.model, num_classes=len(classes), pretrained=args.pretrained, cfg=cfg)
    if args.freeze_backbone:
        model.freeze_backbone(True)
    if args.grad_ckpt:
        model.set_grad_checkpointing(True)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model {args.model}: {n_params / 1e6:.1f}M params ({n_train / 1e6:.1f}M trainable)")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args)
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    steps_per_epoch = math.ceil((args.max_train_steps or len(loaders["train"])) / args.accum_steps)
    state = {"step": 0, "total_steps": steps_per_epoch * args.epochs,
             "warmup_steps": int(steps_per_epoch * args.warmup_epochs)}
    start_epoch, best_acc, history = 1, 0.0, []

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        state = ck["train_state"]
        start_epoch, best_acc, history = ck["epoch"] + 1, ck["best_acc"], ck.get("history", [])
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best {best_acc:.2f}%)")

    writer = SummaryWriter(os.path.join(args.log_dir, run))
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "model_cfg": cfg.to_dict(), "classes": classes}, f, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, device,
                                          amp_dtype, epoch, args, state, writer)
        va_loss, va_top1, va_top5 = evaluate(model, loaders["val"], criterion, device, amp_dtype,
                                             max_batches=args.max_val_batches)
        dt = time.time() - t0
        rec = dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss,
                   val_top1=va_top1, val_top5=va_top5, seconds=dt)
        history.append(rec)
        for k, v in rec.items():
            if k != "epoch":
                writer.add_scalar(f"epoch/{k}", v, epoch)
        print(f"Epoch {epoch}/{args.epochs} ({dt / 60:.1f} min) | train loss {tr_loss:.4f} acc {tr_acc:.2f}% "
              f"| val loss {va_loss:.4f} top1 {va_top1:.2f}% top5 {va_top5:.2f}%")

        metrics = {"val_top1": va_top1, "val_top5": va_top5, "val_loss": va_loss}
        if va_top1 > best_acc:
            best_acc = va_top1
            save_checkpoint(os.path.join(ckpt_dir, "best.pth"), model, classes, epoch, metrics, vars(args))
            print(f"  new best: {best_acc:.2f}% -> {ckpt_dir}/best.pth")
        save_checkpoint(os.path.join(ckpt_dir, "last.pth"), model, classes, epoch, metrics, vars(args),
                        extra={"optimizer_state_dict": optimizer.state_dict(), "train_state": state,
                               "best_acc": best_acc, "history": history})
        with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    writer.close()
    print(f"Done. Best val top-1: {best_acc:.2f}%  (checkpoint: {ckpt_dir}/best.pth)")


if __name__ == "__main__":
    main()
