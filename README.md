# YOLO11 with ResNet + FPN + Graph Attention

This project implements a YOLO11-style object detector with three key upgrades:

- **Backbone:** configurable ResNet (18/34/50) for hierarchical feature extraction.
- **Neck:** Feature Pyramid Network (FPN) to aggregate multi-scale features.
- **Enhancement:** Graph Attention modules (GAT) on the S2/S3 pyramid levels before prediction.

It ships with training, validation, and inference scripts plus a configurable YAML-driven pipeline.

## Folder Layout

```
WorkSpace/
├── config.yaml                 # Default hyper-parameter configuration
├── requirements.txt            # Runtime & tooling dependencies
├── train.py / val.py / predict.py
├── yolo11/
│   ├── __init__.py
│   ├── models/                 # Backbone, FPN, GAT, detection head, losses
│   └── utils/                  # Config loader, dataset helper, decoding
└── tests/test_model.py         # Smoke tests for model + loss
```

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Prepare your dataset in YOLO格式。若已存在 `train/`、`val/` 两个子目录，可直接指向 `data.train_dir` 与 `data.val_dir`。如果只有一个平铺目录（如 `ship_dataset_v0`，图像与标签同级摆放），将该路径填入 `data.source_dir`，首次训练时会自动打乱并按 `data.val_split` 划分为 `train/`、`val/` 两个子目录（默认 80% 训练 / 20% 验证）。

每个标签文件需包含如下格式：

```
<class_id> <x_center> <y_center> <width> <height>
```

Values are normalized to `[0, 1]` with respect to image width/height. Update the paths in `config.yaml` (`data.train_dir`, `data.val_dir`) to point at your dataset splits.

## Training

```powershell
python train.py --config config.yaml
```

Key configuration knobs (from `config.yaml`):

- `model.num_classes`: number of detection classes.
- `model.backbone_depth`: choose 18, 34, or 50.
- `data.source_dir`: flat dataset root，用于自动划分；留空则跳过该步骤。
- `data.val_split`: 验证集占比（0-1）。
- `runtime.epochs`, `runtime.device`, `runtime.checkpoint_dir`.

Checkpoints are written to `./checkpoints/epoch_XXX.pt`. Resume training with `--resume path/to/checkpoint.pt`.

## Validation

```powershell
python val.py --config config.yaml --checkpoint checkpoints/epoch_100.pt
```

Outputs the average loss on the validation split and the mean number of detections per image. Adjust confidence / NMS thresholds using `--conf` and `--nms`.

## Inference

```powershell
python predict.py --config config.yaml --checkpoint checkpoints/epoch_100.pt --image path/to/image.jpg --output runs/preds
```

You can also specify `--input-dir` for batch inference. Predictions are saved as JSON files with bounding boxes (in pixel coordinates), scores, and class IDs. The helper also supports `--conf` and `--nms` overrides for post-processing.

## Testing

```powershell
python -m pytest
```

The smoke tests cover:

- Forward shape checks for the YOLO11 model on dummy tensors.
- End-to-end loss computation with backprop to ensure gradients flow.

## References

- Kaiming He et al., *Deep Residual Learning for Image Recognition*. CVPR 2016. https://arxiv.org/abs/1512.03385
- Tsung-Yi Lin et al., *Feature Pyramid Networks for Object Detection*. CVPR 2017. https://arxiv.org/abs/1612.03144
- Petar Veličković et al., *Graph Attention Networks*. ICLR 2018. https://arxiv.org/abs/1710.10903

## Roadmap / Ideas

- Implement stronger label assignment (e.g., SimOTA) and IoU-aware losses.
- Add mixed-precision-friendly exponential moving average (EMA) weights.
- Integrate richer augmentation (mosaic/mixup) and evaluation metrics (mAP@0.5:0.95).

## Desktop App Wrapper (Windows Target)

This repository now includes a desktop application wrapper for the model:

- Entry: `run_app.py`
- UI: `app/main_window.py`
- Inference service: `app/infer_engine.py`

### Launch App

```powershell
pip install -r requirements.txt
python run_app.py
```

### App Features

- Load model config + checkpoint
- Single-image detection with visual overlay
- Batch detection for a folder
- Export per-image JSON
- Export batch CSV summary
- Runtime logs in UI

### Build Windows EXE (PyInstaller)

```powershell
pip install pyinstaller
pyinstaller --noconfirm --windowed --name SmartDetectionApp run_app.py
```

After build, executable is generated under `dist/SmartDetectionApp/`.
