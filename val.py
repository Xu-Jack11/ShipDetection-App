from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolo11 import YOLO11
from yolo11.models import YOLODetectionLoss
from yolo11.utils import (
    ProjectConfig,
    YOLODetectionDataset,
    decode_predictions,
    detection_collate_fn,
    ensure_dataset_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the YOLO11 detector")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file for evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for decoding predictions")
    parser.add_argument("--nms", type=float, default=0.5, help="NMS IoU threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig.from_yaml(args.config)

    if args.device is not None:
        cfg.runtime.device = args.device

    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")

    model = YOLO11(
        num_classes=cfg.model.num_classes,
        backbone_depth=cfg.model.backbone_depth,
        fpn_channels=cfg.model.fpn_channels,
        gat_levels=cfg.model.gat_levels,
        gat_heads=cfg.model.gat_heads,
    ).to(device)

    checkpoint_path = Path(args.checkpoint)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    ensure_dataset_split(cfg.data)

    val_dataset = YOLODetectionDataset(cfg.data.val_dir, image_size=cfg.data.image_size, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
    )

    criterion = YOLODetectionLoss(num_classes=cfg.model.num_classes).to(device)

    accumulated_loss = 0.0
    total_batches = 0
    detections = 0
    total_images = 0

    with torch.no_grad():
        for images, targets, _, orig_sizes in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            targets_on_device = [
                {
                    "boxes": target["boxes"].to(device),
                    "labels": target["labels"].to(device),
                }
                for target in targets
            ]
            predictions = model(images)
            loss = criterion(predictions, targets_on_device)

            decoded = decode_predictions(predictions, cfg.model.num_classes, args.conf, args.nms, orig_sizes)
            detections += sum(det["boxes"].shape[0] for det in decoded)
            total_images += images.size(0)

            accumulated_loss += loss.item()
            total_batches += 1

    avg_loss = accumulated_loss / max(total_batches, 1)
    avg_det = detections / max(total_images, 1)

    print(f"Validation loss: {avg_loss:.4f}")
    print(f"Average detections per image: {avg_det:.2f}")


if __name__ == "__main__":
    main()
