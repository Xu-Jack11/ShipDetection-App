from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ModelConfig:
    num_classes: int = 1
    backbone_depth: int = 50
    fpn_channels: int = 256
    gat_levels: List[str] = field(default_factory=lambda: ["s2", "s3"])
    gat_heads: int = 4


@dataclass
class OptimConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    warmup_epochs: int = 3
    cosine_anneal: bool = True


@dataclass
class DataConfig:
    train_dir: str = "./data/train"
    val_dir: str = "./data/val"
    image_size: int = 640
    num_workers: int = 4
    batch_size: int = 8
    source_dir: str | None = None
    val_split: float = 0.2
    shuffle_seed: int = 42


@dataclass
class RuntimeConfig:
    epochs: int = 100
    device: str = "cuda"
    amp: bool = True
    checkpoint_dir: str = "./checkpoints"
    resume: str | None = None


@dataclass
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        def build_section(section_name: str, dataclass_type):
            section_data = raw.get(section_name, {})
            return dataclass_type(**section_data)

        return cls(
            model=build_section("model", ModelConfig),
            optimizer=build_section("optimizer", OptimConfig),
            data=build_section("data", DataConfig),
            runtime=build_section("runtime", RuntimeConfig),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": vars(self.model),
            "optimizer": vars(self.optimizer),
            "data": vars(self.data),
            "runtime": vars(self.runtime),
        }
