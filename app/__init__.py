"""Desktop application wrapper for YOLO11 detector."""

from .infer_engine import Detection, DetectionEngine

__all__ = ["Detection", "DetectionEngine"]
