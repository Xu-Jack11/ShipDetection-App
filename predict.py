from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms

from yolo11 import YOLO11
from yolo11.utils import ProjectConfig, decode_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the YOLO11 detector")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file for inference")
    parser.add_argument("--image", type=str, default=None, help="Single image path for inference")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory of images for batch inference")
    parser.add_argument("--output", type=str, default="predictions", help="Directory to store prediction JSON files")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.5, help="NMS IoU threshold")
    return parser.parse_args()


def gather_images(image_path: str | None, input_dir: str | None) -> List[Path]:
    paths: List[Path] = []
    if image_path:
        paths.append(Path(image_path))
    if input_dir:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(Path(input_dir).glob(ext))
    if not paths:
        raise ValueError("No images provided for inference")
    return sorted(paths)


def load_image(path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    orig_w, orig_h = image.size
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor, (orig_h, orig_w)


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
    state = torch.load(Path(args.checkpoint), map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    images = gather_images(args.image, args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        image_tensor, orig_size = load_image(image_path, cfg.data.image_size)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            predictions = model(image_tensor)
        decoded = decode_predictions(predictions, cfg.model.num_classes, args.conf, args.nms, [orig_size])
        det = decoded[0]

        result = {
            "image": str(image_path),
            "detections": [
                {
                    "box": det["boxes"][i].tolist(),
                    "score": float(det["scores"][i].item()),
                    "label": int(det["labels"][i].item()),
                }
                for i in range(det["boxes"].shape[0])
            ],
        }

        output_file = output_dir / f"{image_path.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved predictions for {image_path} -> {output_file}")


if __name__ == "__main__":
    main()
