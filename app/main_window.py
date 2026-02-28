from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .infer_engine import DetectionEngine


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("智能目标识别分析系统 V1.0")
        self.resize(1280, 820)

        self.engine: DetectionEngine | None = None
        self.current_result: dict | None = None
        self.batch_results: list[dict] = []

        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        main = QHBoxLayout(root)
        left = QVBoxLayout()
        right = QVBoxLayout()

        main.addLayout(left, 2)
        main.addLayout(right, 3)

        cfg_group = QGroupBox("模型配置")
        cfg_grid = QGridLayout(cfg_group)

        self.config_input = QLineEdit("config.yaml")
        self.ckpt_input = QLineEdit("checkpoints/epoch_010.pt")
        self.class_input = QLineEdit("ship")

        btn_cfg = QPushButton("选择配置")
        btn_ckpt = QPushButton("选择权重")
        btn_cfg.clicked.connect(self.choose_config)
        btn_ckpt.clicked.connect(self.choose_checkpoint)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)

        self.nms_spin = QDoubleSpinBox()
        self.nms_spin.setRange(0.01, 1.0)
        self.nms_spin.setSingleStep(0.05)
        self.nms_spin.setValue(0.50)

        self.btn_load = QPushButton("加载模型")
        self.btn_image = QPushButton("单图识别")
        self.btn_folder = QPushButton("批量识别")
        self.btn_export_json = QPushButton("导出当前JSON")
        self.btn_export_csv = QPushButton("导出批量CSV")

        self.btn_load.clicked.connect(self.load_model)
        self.btn_image.clicked.connect(self.detect_single_image)
        self.btn_folder.clicked.connect(self.detect_folder)
        self.btn_export_json.clicked.connect(self.export_current_json)
        self.btn_export_csv.clicked.connect(self.export_batch_csv)

        cfg_grid.addWidget(QLabel("配置文件"), 0, 0)
        cfg_grid.addWidget(self.config_input, 0, 1)
        cfg_grid.addWidget(btn_cfg, 0, 2)

        cfg_grid.addWidget(QLabel("权重文件"), 1, 0)
        cfg_grid.addWidget(self.ckpt_input, 1, 1)
        cfg_grid.addWidget(btn_ckpt, 1, 2)

        cfg_grid.addWidget(QLabel("类别名称"), 2, 0)
        cfg_grid.addWidget(self.class_input, 2, 1, 1, 2)

        cfg_grid.addWidget(QLabel("置信度阈值"), 3, 0)
        cfg_grid.addWidget(self.conf_spin, 3, 1)
        cfg_grid.addWidget(QLabel("NMS阈值"), 3, 2)
        cfg_grid.addWidget(self.nms_spin, 3, 3)

        cfg_grid.addWidget(self.btn_load, 4, 0)
        cfg_grid.addWidget(self.btn_image, 4, 1)
        cfg_grid.addWidget(self.btn_folder, 4, 2)

        cfg_grid.addWidget(self.btn_export_json, 5, 0, 1, 2)
        cfg_grid.addWidget(self.btn_export_csv, 5, 2, 1, 2)

        left.addWidget(cfg_group)

        self.preview = QLabel("识别结果预览")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(420)
        self.preview.setStyleSheet("border: 1px solid #999; background: #1f1f1f; color: #ddd;")
        left.addWidget(self.preview)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["类别", "置信度", "x1", "y1", "x2", "y2", "图片"])
        right.addWidget(self.table)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("运行日志")
        right.addWidget(self.log)

    def _append_log(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {text}")

    def choose_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", str(Path.cwd()), "YAML (*.yaml *.yml)")
        if path:
            self.config_input.setText(path)

    def choose_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择权重文件", str(Path.cwd()), "PyTorch Checkpoint (*.pt *.pth)")
        if path:
            self.ckpt_input.setText(path)

    def _ensure_engine(self) -> DetectionEngine:
        if self.engine is None:
            raise RuntimeError("模型未加载，请先点击‘加载模型’")
        return self.engine

    def load_model(self) -> None:
        try:
            class_names = [x.strip() for x in self.class_input.text().split(",") if x.strip()]
            if not class_names:
                class_names = ["ship"]

            config_path = Path(self.config_input.text().strip())
            checkpoint_path = Path(self.ckpt_input.text().strip())

            self.engine = DetectionEngine(config_path=config_path, class_names=class_names)
            self.engine.load_model(checkpoint_path)
            self._append_log(f"模型加载成功: {checkpoint_path}")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))
            self._append_log(f"模型加载失败: {e}")

    def detect_single_image(self) -> None:
        try:
            engine = self._ensure_engine()
            image_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择图片",
                str(Path.cwd()),
                "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
            )
            if not image_path:
                return

            result = engine.predict_image(image_path, conf=self.conf_spin.value(), nms=self.nms_spin.value())
            self.current_result = result
            self._append_log(f"单图识别完成: {Path(image_path).name}, 检测数={len(result['detections'])}")

            vis = engine.draw_detections(image_path, result["detections"])
            qimg = ImageQt(vis)
            pixmap = QPixmap.fromImage(qimg)
            self.preview.setPixmap(pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            self._fill_table([result])
        except Exception as e:
            QMessageBox.critical(self, "识别失败", str(e))
            self._append_log(f"单图识别失败: {e}")

    def detect_folder(self) -> None:
        try:
            engine = self._ensure_engine()
            folder = QFileDialog.getExistingDirectory(self, "选择图片目录", str(Path.cwd()))
            if not folder:
                return

            results = engine.predict_folder(folder, conf=self.conf_spin.value(), nms=self.nms_spin.value())
            self.batch_results = results
            self._fill_table(results)

            output_root = Path("runs/app_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
            output_root.mkdir(parents=True, exist_ok=True)

            for item in results:
                img_name = Path(item["image"]).stem
                json_path = output_root / f"{img_name}.json"
                engine.save_result_json(item, json_path)

            csv_path = output_root / "batch_results.csv"
            engine.save_batch_csv(results, csv_path, engine.class_names)

            self._append_log(f"批量识别完成: {len(results)} 张图，结果目录: {output_root}")
            QMessageBox.information(self, "完成", f"批量识别完成，输出目录:\n{output_root}")
        except Exception as e:
            QMessageBox.critical(self, "批量识别失败", str(e))
            self._append_log(f"批量识别失败: {e}")

    def export_current_json(self) -> None:
        if not self.current_result:
            QMessageBox.information(self, "提示", "当前没有单图识别结果")
            return

        if self.engine is None:
            QMessageBox.information(self, "提示", "模型未加载")
            return

        path, _ = QFileDialog.getSaveFileName(self, "导出当前结果JSON", str(Path.cwd() / "result.json"), "JSON (*.json)")
        if not path:
            return

        self.engine.save_result_json(self.current_result, path)
        self._append_log(f"已导出JSON: {path}")

    def export_batch_csv(self) -> None:
        if not self.batch_results:
            QMessageBox.information(self, "提示", "当前没有批量识别结果")
            return

        if self.engine is None:
            QMessageBox.information(self, "提示", "模型未加载")
            return

        path, _ = QFileDialog.getSaveFileName(self, "导出批量CSV", str(Path.cwd() / "batch_results.csv"), "CSV (*.csv)")
        if not path:
            return

        self.engine.save_batch_csv(self.batch_results, path, self.engine.class_names)
        self._append_log(f"已导出CSV: {path}")

    def _fill_table(self, results: list[dict]) -> None:
        rows: list[tuple] = []
        class_names = self.engine.class_names if self.engine else ["ship"]

        for item in results:
            image = item["image"]
            for det in item.get("detections", []):
                label_id = int(det["label"])
                label = class_names[label_id] if 0 <= label_id < len(class_names) else f"class_{label_id}"
                score = det["score"]
                x1, y1, x2, y2 = det["box"]
                rows.append((label, score, x1, y1, x2, y2, Path(image).name))

        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                if isinstance(value, float):
                    text = f"{value:.3f}"
                else:
                    text = str(value)
                self.table.setItem(r, c, QTableWidgetItem(text))
