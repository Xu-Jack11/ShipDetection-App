from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor
from torchvision.ops import nms


def _generate_grid(height: int, width: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    ys = torch.arange(height, device=device).view(height, 1).expand(height, width)
    xs = torch.arange(width, device=device).view(1, width).expand(height, width)
    return ys, xs


def decode_predictions(
    predictions: List[Tensor],
    num_classes: int,
    conf_thresh: float,
    nms_thresh: float,
    image_sizes: Sequence[Tuple[int, int]],
) -> List[Dict[str, Tensor]]:
    """Decode raw network outputs into bounding boxes per image."""

    batch_size = predictions[0].shape[0]
    device = predictions[0].device

    decoded: List[Dict[str, Tensor]] = []

    combined_boxes: List[List[Tensor]] = [[] for _ in range(batch_size)]
    combined_scores: List[List[Tensor]] = [[] for _ in range(batch_size)]
    combined_labels: List[List[Tensor]] = [[] for _ in range(batch_size)]

    for pred in predictions:
        b, _, h, w = pred.shape
        assert b == batch_size
        grid_y, grid_x = _generate_grid(h, w, device)
        grid_y = grid_y.unsqueeze(0)
        grid_x = grid_x.unsqueeze(0)

        pred = pred.sigmoid()
        box = pred[:, :4]
        obj = pred[:, 4:5]
        cls_scores = pred[:, 5:]

        cx = (grid_x + box[:, 0:1]) / w
        cy = (grid_y + box[:, 1:2]) / h
        bw = torch.clamp(box[:, 2:3], min=1e-6, max=1.0)
        bh = torch.clamp(box[:, 3:4], min=1e-6, max=1.0)

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        x1 = x1.clamp(0.0, 1.0)
        y1 = y1.clamp(0.0, 1.0)
        x2 = x2.clamp(0.0, 1.0)
        y2 = y2.clamp(0.0, 1.0)

        obj = obj.expand_as(cls_scores)
        scores = obj * cls_scores
        scores_flat = scores.reshape(batch_size, -1, num_classes)
        boxes_flat = torch.stack([x1, y1, x2, y2], dim=-1).reshape(batch_size, -1, 4)

        class_scores, labels = scores_flat.max(dim=-1)

        for idx in range(batch_size):
            mask = class_scores[idx] >= conf_thresh
            if mask.sum() == 0:
                continue
            selected_scores = class_scores[idx][mask]
            selected_labels = labels[idx][mask]
            selected_boxes = boxes_flat[idx][mask]

            img_h, img_w = image_sizes[idx]
            scale = torch.tensor([img_w, img_h, img_w, img_h], device=device)
            selected_boxes = selected_boxes * scale

            keep = nms(selected_boxes, selected_scores, nms_thresh)
            combined_boxes[idx].append(selected_boxes[keep])
            combined_scores[idx].append(selected_scores[keep])
            combined_labels[idx].append(selected_labels[keep])

    for idx in range(batch_size):
        if combined_boxes[idx]:
            boxes = torch.cat(combined_boxes[idx], dim=0)
            scores = torch.cat(combined_scores[idx], dim=0)
            labels = torch.cat(combined_labels[idx], dim=0)
        else:
            boxes = torch.zeros((0, 4), device=device)
            scores = torch.zeros((0,), device=device)
            labels = torch.zeros((0,), dtype=torch.long, device=device)
        decoded.append({"boxes": boxes, "scores": scores, "labels": labels})

    return decoded
