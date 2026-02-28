from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolo11 import YOLO11
from yolo11.models import YOLODetectionLoss
from yolo11.utils import (
    ProjectConfig,
    YOLODetectionDataset,
    detection_collate_fn,
    ensure_dataset_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the YOLO11 detector")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g., cuda or cpu")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("epoch", 0)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, path)


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig.from_yaml(args.config)

    if args.device is not None:
        cfg.runtime.device = args.device
    if args.resume is not None:
        cfg.runtime.resume = args.resume

    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")

    # Enable cuDNN autotuner for faster convolutions
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    model = YOLO11(
        num_classes=cfg.model.num_classes,
        backbone_depth=cfg.model.backbone_depth,
        fpn_channels=cfg.model.fpn_channels,
        gat_levels=cfg.model.gat_levels,
        gat_heads=cfg.model.gat_heads,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)

    if cfg.optimizer.cosine_anneal:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.runtime.epochs,
            eta_min=cfg.optimizer.learning_rate * 0.1,
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    scaler = GradScaler(enabled=cfg.runtime.amp and device.type == "cuda")

    ensure_dataset_split(cfg.data)

    train_dataset = YOLODetectionDataset(cfg.data.train_dir, image_size=cfg.data.image_size, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
        collate_fn=detection_collate_fn,
        prefetch_factor=2 if cfg.data.num_workers > 0 else None,
        persistent_workers=cfg.data.num_workers > 0,
    )

    criterion = YOLODetectionLoss(num_classes=cfg.model.num_classes).to(device)

    start_epoch = 0
    if cfg.runtime.resume:
        resume_path = Path(cfg.runtime.resume)
        if resume_path.exists():
            start_epoch = load_checkpoint(model, optimizer, resume_path)
            print(f"Resumed from {resume_path} at epoch {start_epoch}")

    best_loss = float("inf")

    for epoch in range(start_epoch, cfg.runtime.epochs):
        model.train()
        running_loss = 0.0
        processed_samples = 0
        epoch_start = time.time()

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.runtime.epochs}",
            total=len(train_loader),
            leave=False,
            dynamic_ncols=True,
        )

        batch_idx = 0
        for batch_idx, (images, targets, _, _) in enumerate(progress_bar, start=1):
            images = images.to(device)
            targets_on_device = [
                {
                    "boxes": target["boxes"].to(device),
                    "labels": target["labels"].to(device),
                }
                for target in targets
            ]

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda",enabled=scaler.is_enabled()):
                predictions = model(images)
                loss = criterion(predictions, targets_on_device)

            # Check for NaN loss and skip iteration if detected
            if not torch.isfinite(loss):
                tqdm.write(f"[WARN] NaN/Inf loss detected at batch {batch_idx}, skipping...")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            # Gradient clipping to prevent explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            processed_samples += images.size(0)

            elapsed = max(time.time() - epoch_start, 1e-6)
            imgs_per_sec = processed_samples / elapsed
            avg_loss = running_loss / batch_idx
            progress_bar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "lr": optimizer.param_groups[0]["lr"],
                    "img/s": f"{imgs_per_sec:.1f}",
                }
            )

        scheduler.step()

        checkpoint_path = Path(cfg.runtime.checkpoint_dir) / f"epoch_{epoch+1:03d}.pt"
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / max(batch_idx, 1)
        best_loss = min(best_loss, epoch_loss)
        tqdm.write(
            " | ".join(
                [
                    f"Epoch {epoch+1}/{cfg.runtime.epochs}",
                    f"loss: {epoch_loss:.4f}",
                    f"best: {best_loss:.4f}",
                    f"img/s: {processed_samples / max(epoch_time, 1e-6):.1f}",
                    f"time: {epoch_time:.1f}s",
                ]
            )
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
