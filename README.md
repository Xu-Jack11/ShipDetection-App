# 船舶目标检测系统（YOLO11 + ResNet + FPN + 图注意力）

本项目实现了一个面向船舶目标检测的 YOLO11 风格模型，并提供训练、验证、推理和桌面应用封装能力。

核心改进：

- 骨干网络：可配置 ResNet（18/34/50）用于多层特征提取。
- 颈部网络：FPN（特征金字塔）用于多尺度特征融合。
- 注意力增强：在高分辨率特征层引入图注意力模块（GAT）。

此外，项目已集成 Windows 桌面应用（PySide6），支持可视化检测与结果导出。

## 项目结构

```text
WorkSpace/
├── config.yaml                 # 默认配置（模型/训练/数据/运行参数）
├── requirements.txt            # 依赖列表
├── train.py                    # 训练脚本
├── val.py                      # 验证脚本
├── predict.py                  # 推理脚本
├── run_app.py                  # 桌面应用入口
├── app/
│   ├── main_window.py          # 桌面 UI
│   └── infer_engine.py         # 推理引擎封装
├── yolo11/
│   ├── models/                 # backbone/fpn/gat/head/loss
│   └── utils/                  # 配置、数据、解码等工具
└── tests/test_model.py         # 基础测试
```

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 数据集说明

支持 YOLO 标签格式。每个标签文件每行格式如下：

```text
<class_id> <x_center> <y_center> <width> <height>
```

其中坐标是相对于图像宽高归一化到 `[0, 1]` 的数值。

你可以使用两种数据组织方式：

1) 已有 train/val 划分
- `data.train_dir` 指向训练集目录
- `data.val_dir` 指向验证集目录

2) 平铺目录自动划分
- 将原始数据目录填入 `data.source_dir`（如 `./ship_dataset_v0`）
- 首次训练时会按 `data.val_split` 自动划分为 train/val（默认 8:2）

## 模型训练

```powershell
python train.py --config config.yaml
```

常用配置项（`config.yaml`）：

- `model.num_classes`：类别数
- `model.backbone_depth`：18 / 34 / 50
- `data.source_dir`：平铺数据集目录（可选）
- `data.val_split`：验证集比例
- `runtime.epochs`：训练轮数
- `runtime.device`：运行设备（cuda/cpu）
- `runtime.checkpoint_dir`：权重输出目录

训练权重默认保存到：

- `./checkpoints/epoch_XXX.pt`

## 模型验证

```powershell
python val.py --config config.yaml --checkpoint checkpoints/epoch_010.pt
```

可通过参数调整阈值：

- `--conf`：置信度阈值
- `--nms`：NMS IoU 阈值

## 模型推理

单图推理：

```powershell
python predict.py --config config.yaml --checkpoint checkpoints/epoch_010.pt --image path/to/image.jpg --output runs/preds
```

批量推理：

```powershell
python predict.py --config config.yaml --checkpoint checkpoints/epoch_010.pt --input-dir path/to/images --output runs/preds
```

输出为 JSON，包含检测框、分数和类别。

## 测试

```powershell
python -m pytest
```

测试覆盖：

- 模型前向输出形状检查
- 检测损失反向传播可用性检查

## 桌面应用（Windows）

项目内置了桌面应用封装：

- 入口：`run_app.py`
- UI：`app/main_window.py`
- 推理封装：`app/infer_engine.py`

### 直接运行（开发模式）

```powershell
pip install -r requirements.txt
python run_app.py
```

### 已实现功能

- 加载配置文件与模型权重
- 单图识别（可视化框选）
- 批量识别（目录）
- 导出单图 JSON
- 导出批量 CSV
- 运行日志显示

## Windows 预构建包下载

仓库已配置 GitHub Actions 自动构建 Windows 可执行文件。

- 工作流页面：
  - https://github.com/Xu-Jack11/ShipDetection-App/actions/workflows/windows-build.yml
- 运行记录：
  - https://github.com/Xu-Jack11/ShipDetection-App/actions

下载步骤：

1. 打开 Actions 页面。
2. 进入最新一次成功的 `Build Windows App` 任务。
3. 下载产物 `SmartDetectionApp-windows`。
4. 解压后运行 `SmartDetectionApp.exe`。

## 在本地 Windows 构建 EXE

```powershell
# 1) 进入项目目录
cd ShipDetection-App

# 2) 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\activate

# 3) 安装依赖
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

# 4) 打包
pyinstaller --noconfirm --clean --windowed --name SmartDetectionApp run_app.py
```

构建完成后产物位置：

- `dist/SmartDetectionApp/SmartDetectionApp.exe`
- 可选手动压缩包：`dist/SmartDetectionApp-windows.zip`

## Windows 首次运行说明

- 若出现 SmartScreen 提示，可点击 `More info` → `Run anyway`（未签名应用常见提示）。
- 请提前准备好 `config.yaml` 与模型权重 `.pt` 文件。
- 在应用内先加载配置和权重，再进行单图/批量识别。

## 参考文献

- Kaiming He et al., Deep Residual Learning for Image Recognition, CVPR 2016
- Tsung-Yi Lin et al., Feature Pyramid Networks for Object Detection, CVPR 2017
- Petar Veličković et al., Graph Attention Networks, ICLR 2018
