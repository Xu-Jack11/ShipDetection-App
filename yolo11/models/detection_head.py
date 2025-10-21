from __future__ import annotations

from typing import Dict, List

from torch import Tensor, nn


class DetectionHead(nn.Module):
    """Anchor-free detection head that predicts bounding boxes per location."""

    def __init__(
        self,
        in_channels: Dict[str, int],
        num_classes: int,
        shared_channels: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.scales = list(in_channels.keys())
        self.branches = nn.ModuleDict()

        for name, channels in in_channels.items():
            self.branches[name] = nn.Sequential(
                nn.Conv2d(channels, shared_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(shared_channels, shared_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(shared_channels, num_classes + 5, kernel_size=1),
            )

        # Initialize final conv layer with small weights to prevent extreme initial predictions
        for branch in self.branches.values():
            final_conv = branch[-1]
            nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
            if final_conv.bias is not None:
                nn.init.constant_(final_conv.bias, 0.0)

    def forward(self, features: Dict[str, Tensor]) -> List[Tensor]:
        outputs = []
        for name in self.scales:
            outputs.append(self.branches[name](features[name]))
        return outputs
