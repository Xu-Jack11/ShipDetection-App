from __future__ import annotations

import torch

from yolo11 import YOLO11
from yolo11.models import YOLODetectionLoss


def test_yolo11_forward_shapes():
    model = YOLO11(num_classes=3, backbone_depth=18, fpn_channels=128)
    model.eval()

    dummy = torch.randn(2, 3, 256, 256)
    outputs = model(dummy)

    assert len(outputs) == 3
    for tensor in outputs:
        assert tensor.shape[0] == 2
        assert tensor.shape[1] == 3 + 5

    strides = [4, 8, 16]
    for out, stride in zip(outputs, strides):
        expected = 256 // stride
        assert out.shape[2] == expected
        assert out.shape[3] == expected


def test_yolo_detection_loss_backward():
    model = YOLO11(num_classes=2, backbone_depth=18, fpn_channels=64)
    criterion = YOLODetectionLoss(num_classes=2)

    dummy = torch.randn(2, 3, 256, 256, requires_grad=True)
    outputs = model(dummy)

    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
        },
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        },
    ]

    loss = criterion(outputs, targets)
    loss.backward()

    assert loss.item() >= 0
