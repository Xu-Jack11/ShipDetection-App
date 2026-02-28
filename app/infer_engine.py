from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from yolo11 import YOLO11
from yolo11.utils import ProjectConfig, decode_predictions

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Detection:
    label: int
    score: float
    box: list[float]  # [x1, y1, x2, y2]


class DetectionEngine:
    def __init__(self, config_path: str | Path, class_names: Sequence[str]) -> None:
        self.config_path = Path(config_path)
        self.cfg = ProjectConfig.from_yaml(self.config_path)
        self.class_names = list(class_names)
        self.device = torch.device(self.cfg.runtime.device if torch.cuda.is_available() else "cpu")

        self.model: YOLO11 | None = None
        self.checkpoint_path: Path | None = None
        self._transform = transforms.Compose(
            [
                transforms.Resize((self.cfg.data.image_size, self.cfg.data.image_size)),
                transforms.ToTensor(),
            ]
        )

    def load_model(self, checkpoint_path: str | Path) -> None:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        model = YOLO11(
            num_classes=self.cfg.model.num_classes,
            backbone_depth=self.cfg.model.backbone_depth,
            fpn_channels=self.cfg.model.fpn_channels,
            gat_levels=self.cfg.model.gat_levels,
            gat_heads=self.cfg.model.gat_heads,
        ).to(self.device)

        state = torch.load(ckpt, map_location="cpu")
        if "model" not in state:
            raise ValueError("Checkpoint format invalid: missing key 'model'")

        model.load_state_dict(state["model"])
        model.eval()

        self.model = model
        self.checkpoint_path = ckpt

    def _ensure_model_loaded(self) -> YOLO11:
        if self.model is None:
            raise RuntimeError("Model has not been loaded. Please load checkpoint first.")
        return self.model

    def predict_image(self, image_path: str | Path, conf: float = 0.25, nms: float = 0.5) -> dict:
        model = self._ensure_model_loaded()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path).convert("RGB")
        orig_w, orig_h = image.size
        image_tensor = self._transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = model(image_tensor)

        decoded = decode_predictions(
            predictions,
            num_classes=self.cfg.model.num_classes,
            conf_thresh=conf,
            nms_thresh=nms,
            image_sizes=[(orig_h, orig_w)],
        )[0]

        detections: list[Detection] = []
        for i in range(decoded["boxes"].shape[0]):
            detections.append(
                Detection(
                    label=int(decoded["labels"][i].item()),
                    score=float(decoded["scores"][i].item()),
                    box=[float(v) for v in decoded["boxes"][i].tolist()],
                )
            )

        return {
            "image": str(path),
            "detections": [asdict(det) for det in detections],
            "meta": {
                "config": str(self.config_path),
                "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
                "device": str(self.device),
                "conf": conf,
                "nms": nms,
            },
        }

    def predict_folder(self, folder: str | Path, conf: float = 0.25, nms: float = 0.5) -> list[dict]:
        root = Path(folder)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Input folder not found: {root}")

        image_paths = sorted([p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in {root}")

        return [self.predict_image(p, conf=conf, nms=nms) for p in image_paths]

    def draw_detections(self, image_path: str | Path, detections: Iterable[dict]) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            score = det["score"]
            label_idx = det["label"]
            label_name = self.class_names[label_idx] if 0 <= label_idx < len(self.class_names) else f"class_{label_idx}"
            text = f"{label_name} {score:.2f}"

            draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)
            tx = x1
            ty = max(0, y1 - 18)
            draw.rectangle([tx, ty, tx + 180, ty + 18], fill=(255, 80, 80))
            draw.text((tx + 3, ty + 2), text, fill=(255, 255, 255))

        return image

    @staticmethod
    def save_result_json(result: dict, output_path: str | Path) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def save_batch_csv(results: Sequence[dict], output_path: str | Path, class_names: Sequence[str]) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "label_id", "label_name", "score", "x1", "y1", "x2", "y2"])
            for item in results:
                image = item["image"]
                for det in item.get("detections", []):
                    label_id = int(det["label"])
                    label_name = class_names[label_id] if 0 <= label_id < len(class_names) else f"class_{label_id}"
                    x1, y1, x2, y2 = det["box"]
                    writer.writerow([image, label_id, label_name, f"{det['score']:.6f}", x1, y1, x2, y2])
