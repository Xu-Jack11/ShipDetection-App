from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from .config import DataConfig


class YOLODetectionDataset(Dataset):
    """Simple dataset for YOLO-style annotations.

    The expected directory layout is:

    root/
        images/
            xxx.jpg
        labels/
            xxx.txt  # cls x_center y_center width height (normalized)
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 640,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.label_dir = self.root / "labels"
        self.image_paths = sorted(self.image_dir.glob("*.*"))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        self.image_size = image_size
        self.augment = augment
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_label_file(self, image_path: Path) -> Tensor:
        label_path = self.label_dir / f"{image_path.stem}.txt"
        boxes: List[List[float]] = []
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, xc, yc, w, h = map(float, parts)
                    boxes.append([cls, xc, yc, w, h])
        if not boxes:
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

    def _apply_augment(self, image: Tensor, boxes: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.augment or boxes.numel() == 0:
            return image, boxes
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            boxes = boxes.clone()
            boxes[:, 1] = 1.0 - boxes[:, 1]
        return image, boxes

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        path = self.image_paths[idx]
        image_raw = Image.open(path).convert("RGB")
        orig_w, orig_h = image_raw.size
        image = self.base_transform(image_raw)

        boxes = self._read_label_file(path)
        image, boxes = self._apply_augment(image, boxes)

        cls_labels = boxes[:, 0].long() if boxes.numel() else torch.zeros((0,), dtype=torch.long)
        target_boxes = boxes[:, 1:]

        target = {
            "labels": cls_labels,
            "boxes": target_boxes,
        }
        return {"image": image, "target": target, "path": str(path), "orig_size": (orig_h, orig_w)}


def detection_collate_fn(batch: List[Dict[str, Tensor]]):
    images = torch.stack([item["image"] for item in batch])
    targets = [item["target"] for item in batch]
    paths = [item["path"] for item in batch]
    orig_sizes = [item["orig_size"] for item in batch]
    return images, targets, paths, orig_sizes


def ensure_dataset_split(data_cfg: DataConfig) -> None:
    """Split a flat dataset into train/val splits if they are empty."""

    def _has_images(directory: Path) -> bool:
        return directory.exists() and any(directory.glob("*.*"))

    train_images_dir = Path(data_cfg.train_dir) / "images"
    val_images_dir = Path(data_cfg.val_dir) / "images"

    # Short-circuit when both splits already contain data
    if _has_images(train_images_dir) and _has_images(val_images_dir):
        return

    if data_cfg.source_dir is None:
        raise FileNotFoundError(
            "Train/val directories are empty but data.source_dir is not configured."
        )

    source_dir = Path(data_cfg.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")

    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = [p for p in source_dir.iterdir() if p.suffix.lower() in allowed_suffixes]
    if not image_files:
        raise FileNotFoundError(f"No images found in source directory: {source_dir}")

    rng = random.Random(data_cfg.shuffle_seed)
    rng.shuffle(image_files)

    val_count = max(1, int(len(image_files) * data_cfg.val_split))
    val_files = image_files[:val_count]
    train_files = image_files[val_count:]
    if not train_files:
        train_files, val_files = val_files, []

    for subset_dir in [train_images_dir, val_images_dir]:
        subset_dir.mkdir(parents=True, exist_ok=True)
        (subset_dir.parent / "labels").mkdir(parents=True, exist_ok=True)

    def _copy_pair(image_path: Path, destination_root: Path) -> None:
        labels_dir = destination_root / "labels"
        images_dir = destination_root / "images"
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            print(f"[WARN] Missing label for {image_path.name}, skipping.")
            return
        shutil.copy2(image_path, images_dir / image_path.name)
        shutil.copy2(label_path, labels_dir / label_path.name)

    for image_path in train_files:
        _copy_pair(image_path, train_images_dir.parent)

    for image_path in val_files:
        _copy_pair(image_path, val_images_dir.parent)

    print(
        f"Prepared dataset split: {len(train_files)} training images, {len(val_files)} validation images from {source_dir}."
    )
