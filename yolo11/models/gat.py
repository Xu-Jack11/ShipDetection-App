from __future__ import annotations

from typing import Dict

import torch.nn.functional as F
from torch import Tensor, nn


class SpatialGraphAttention(nn.Module):
    """Graph-attention style block operating on flattened spatial tokens."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        reduce_factor: int = 1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )
        self.reduce_factor = max(1, reduce_factor)
        self.pool = (
            nn.AvgPool2d(kernel_size=self.reduce_factor, stride=self.reduce_factor, ceil_mode=True)
            if self.reduce_factor > 1
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        spatial_residual = x
        pooled = self.pool(x) if self.pool is not None else x

        b, c, h, w = pooled.shape
        tokens = pooled.flatten(2).transpose(1, 2)  # (b, hw, c)

        token_residual = tokens
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(token_residual + attn_out)

        token_residual = tokens
        tokens = self.norm2(token_residual + self.ffn(tokens))

        out = tokens.transpose(1, 2).reshape(b, c, h, w)
        if self.pool is not None:
            out = F.interpolate(out, size=spatial_residual.shape[-2:], mode="bilinear", align_corners=False)

        return spatial_residual + out


class GraphAttentionEnhancer(nn.Module):
    """Applies graph attention to selected feature levels."""

    def __init__(
        self,
        channels_per_level: Dict[str, int],
        heads: int = 4,
        dropout: float = 0.1,
        reductions: Dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        reductions = reductions or {}
        self.blocks = nn.ModuleDict(
            {
                level: SpatialGraphAttention(
                    ch,
                    num_heads=heads,
                    dropout=dropout,
                    reduce_factor=reductions.get(level, 1),
                )
                for level, ch in channels_per_level.items()
            }
        )

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        enhanced = {}
        for name, feat in features.items():
            if name in self.blocks:
                enhanced[name] = self.blocks[name](feat)
            else:
                enhanced[name] = feat
        return enhanced
