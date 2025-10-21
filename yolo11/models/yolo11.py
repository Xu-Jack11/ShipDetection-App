from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

from torch import Tensor, nn

from .backbone import ResNetBackbone, build_resnet_backbone
from .detection_head import DetectionHead
from .fpn import FeaturePyramidNetwork
from .gat import GraphAttentionEnhancer


class YOLO11(nn.Module):
    """YOLO11 detector with ResNet backbone, FPN, and graph attention modules."""

    def __init__(
        self,
        num_classes: int,
        backbone_depth: int = 50,
        fpn_channels: int = 256,
        gat_levels: List[str] | None = None,
        gat_heads: int = 4,
        gat_reductions: Dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.backbone: ResNetBackbone = build_resnet_backbone(depth=backbone_depth)

        backbone_channels = self.backbone.out_channels
        pyramid_in_channels = OrderedDict((level, backbone_channels[level]) for level in ["s2", "s3", "s4", "s5"])
        self.fpn = FeaturePyramidNetwork(pyramid_in_channels, fpn_channels)

        gat_levels = gat_levels or ["s2", "s3"]
        self.gat_levels = gat_levels
        default_reductions = {lvl: 1 for lvl in gat_levels}
        if "s2" in default_reductions:
            default_reductions["s2"] = 4
        if "s3" in default_reductions:
            default_reductions["s3"] = 2
        if gat_reductions:
            default_reductions.update(gat_reductions)

        self.graph_attention = GraphAttentionEnhancer(
            {lvl: fpn_channels for lvl in gat_levels},
            heads=gat_heads,
            reductions=default_reductions,
        )

        head_channels = {name: fpn_channels for name in ["s2", "s3", "s4"]}
        self.head = DetectionHead(head_channels, num_classes=num_classes)

    def forward(self, x: Tensor) -> List[Tensor]:
        features = self.backbone(x)

        # Only keep levels configured for the FPN
        pyramid_inputs = {k: features[k] for k in ["s2", "s3", "s4", "s5"]}
        pyramid_feats = self.fpn(pyramid_inputs)

        enhanced = self.graph_attention({lvl: pyramid_feats[lvl] for lvl in self.gat_levels})
        pyramid_feats = pyramid_feats.copy()
        for lvl, feat in enhanced.items():
            pyramid_feats[lvl] = feat

        head_inputs = {lvl: pyramid_feats[lvl] for lvl in self.head.scales}
        outputs = self.head(head_inputs)
        return outputs
