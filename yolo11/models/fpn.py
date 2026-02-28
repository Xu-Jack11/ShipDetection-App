from __future__ import annotations

from typing import Dict, Iterable

from torch import Tensor, nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    """Top-down feature pyramid network with lateral connections."""

    def __init__(self, in_channels: Dict[str, int], out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.levels = list(in_channels.keys())

        self.lateral_convs = nn.ModuleDict()
        self.output_convs = nn.ModuleDict()

        for name, c in in_channels.items():
            self.lateral_convs[name] = nn.Conv2d(c, out_channels, kernel_size=1)
            self.output_convs[name] = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        assert set(inputs.keys()) == set(
            self.levels
        ), f"Expected feature keys {self.levels}, got {list(inputs.keys())}"

        results: Dict[str, Tensor] = {}
        last_inner: Tensor | None = None

        for level in reversed(self.levels):
            feat = inputs[level]
            lateral = self.lateral_convs[level](feat)
            if last_inner is not None:
                lateral = lateral + F.interpolate(last_inner, size=lateral.shape[-2:], mode="nearest")
            results[level] = self.output_convs[level](lateral)
            last_inner = lateral

        return results
