from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor, nn


class YOLODetectionLoss(nn.Module):
    """Simplified YOLO-style detection loss.

    This implementation assigns each target to the closest grid cell on each feature map
    and computes a combination of MSE for bounding boxes and BCE for objectness/classification.
    """

    def __init__(self, num_classes: int, lambda_box: float = 5.0, lambda_obj: float = 1.0, lambda_cls: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.mse = nn.MSELoss(reduction="mean")
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, predictions: List[Tensor], targets: List[Dict[str, Tensor]]) -> Tensor:
        total_loss = predictions[0].new_tensor(0.0)

        for pred in predictions:
            b, _, h, w = pred.shape
            obj_target = torch.zeros((b, 1, h, w), device=pred.device)
            box_target = torch.zeros((b, 4, h, w), device=pred.device)
            cls_target = torch.zeros((b, self.num_classes, h, w), device=pred.device)

            for batch_idx, target in enumerate(targets):
                boxes = target["boxes"]
                labels = target["labels"]
                if boxes.numel() == 0:
                    continue
                for box, label in zip(boxes, labels):
                    gx = box[0] * w
                    gy = box[1] * h
                    gw = box[2]
                    gh = box[3]
                    gi = min(int(gx), w - 1)
                    gj = min(int(gy), h - 1)
                    obj_target[batch_idx, 0, gj, gi] = 1.0
                    box_target[batch_idx, 0, gj, gi] = gx - gi
                    box_target[batch_idx, 1, gj, gi] = gy - gj
                    box_target[batch_idx, 2, gj, gi] = gw
                    box_target[batch_idx, 3, gj, gi] = gh
                    cls_target[batch_idx, label, gj, gi] = 1.0

            pred_box = pred[:, :4]
            pred_obj = pred[:, 4:5]
            pred_cls = pred[:, 5:]

            # Clamp predictions to prevent extreme values
            pred_box = torch.clamp(pred_box, min=-10.0, max=10.0)
            pred_obj = torch.clamp(pred_obj, min=-10.0, max=10.0)
            pred_cls = torch.clamp(pred_cls, min=-10.0, max=10.0)

            box_loss = self.mse(pred_box, box_target)
            obj_loss = self.bce(pred_obj, obj_target)
            cls_loss = self.bce(pred_cls, cls_target)

            # Check for NaN/Inf and replace with zero
            if not torch.isfinite(box_loss):
                box_loss = torch.tensor(0.0, device=pred.device)
            if not torch.isfinite(obj_loss):
                obj_loss = torch.tensor(0.0, device=pred.device)
            if not torch.isfinite(cls_loss):
                cls_loss = torch.tensor(0.0, device=pred.device)

            total_loss = total_loss + self.lambda_box * box_loss + self.lambda_obj * obj_loss + self.lambda_cls * cls_loss

        avg_loss = total_loss / max(len(predictions), 1)
        return torch.clamp(avg_loss, min=0.0, max=1e6)
