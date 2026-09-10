from multiprocessing import process
from PyQt5 import QtWidgets, QtCore, QtGui, QtChart
from labelme.utils import fit_dialog
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread
import subprocess
import sys
from PyQt5 import QtWidgets
import os
import re
from collections import deque
import shutil
from matplotlib.table import table
from pyparsing import col, line
import yaml
import glob

def validate_yolo_dataset(dataset_path):
    """
    Validates YOLO dataset structure before training.
    
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    if not dataset_path or not os.path.exists(dataset_path):
        return False, "Dataset path does not exist."
    
    errors = []
    
    # Check for data.yaml
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml_path):
        errors.append("❌ Missing 'data.yaml' in dataset root")
    else:
        # Validate data.yaml contents
        try:
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            required_keys = ['train', 'val', 'nc', 'names']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                errors.append(f"❌ data.yaml missing required keys: {', '.join(missing_keys)}")
            
            if 'nc' in data and 'names' in data:
                if len(data['names']) != data['nc']:
                    errors.append(f"❌ data.yaml: 'nc' ({data['nc']}) doesn't match number of 'names' ({len(data['names'])})")
        except Exception as e:
            errors.append(f"❌ Error reading data.yaml: {str(e)}")
    
    # Check directory structure
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")
    
    if not os.path.exists(images_dir):
        errors.append("❌ Missing 'images' directory")
    if not os.path.exists(labels_dir):
        errors.append("❌ Missing 'labels' directory")
    
    # Check train/val splits
    for split in ['train', 'val']:
        images_split_dir = os.path.join(images_dir, split)
        labels_split_dir = os.path.join(labels_dir, split)
        
        if not os.path.exists(images_split_dir):
            errors.append(f"❌ Missing 'images/{split}' directory")
        else:
            # Check if there are images
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
            images = [f for f in os.listdir(images_split_dir) 
                     if os.path.splitext(f)[1].lower() in image_extensions]
            if len(images) == 0:
                errors.append(f"⚠️ No images found in 'images/{split}' directory")
        
        if not os.path.exists(labels_split_dir):
            errors.append(f"❌ Missing 'labels/{split}' directory")
        else:
            # Check if there are labels
            labels = glob.glob(os.path.join(labels_split_dir, "*.txt"))
            if len(labels) == 0:
                errors.append(f"⚠️ No label files (.txt) found in 'labels/{split}' directory")
    
    if errors:
        error_msg = "Dataset validation failed:\n\n" + "\n".join(errors)
        error_msg += "\n\n📁 Expected structure:\n"
        error_msg += "Dataset/\n"
        error_msg += "├── data.yaml\n"
        error_msg += "├── images/\n"
        error_msg += "│   ├── train/\n"
        error_msg += "│   └── val/\n"
        error_msg += "└── labels/\n"
        error_msg += "    ├── train/\n"
        error_msg += "    └── val/"
        return False, error_msg
    
    return True, None


def save_weights_to_exp_folder(dataset_folder, best_weights_path, last_weights_path):
    weights_dir = os.path.join(dataset_folder, "weight")
    os.makedirs(weights_dir, exist_ok=True)
    # Find next expN folder
    exp_num = 1
    while os.path.exists(os.path.join(weights_dir, f"exp{exp_num}")):
        exp_num += 1
    exp_folder = os.path.join(weights_dir, f"exp{exp_num}")
    os.makedirs(exp_folder, exist_ok=True)
    # Copy weights
    best_dest = os.path.join(exp_folder, "best.pt")
    last_dest = os.path.join(exp_folder, "last.pt")
    shutil.copy2(best_weights_path, best_dest)
    shutil.copy2(last_weights_path, last_dest)
    return best_dest, last_dest

class TrainingResultsDialog(QtWidgets.QDialog):
    def __init__(self, results_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Training Results")
        fit_dialog(self, 900, 600)
        layout = QtWidgets.QVBoxLayout(self)

        # Parse summary info as before...
        epochs = None
        time_taken = None
        last_weights = None
        best_weights = None

        lines = results_text.splitlines()
        for i, line in enumerate(lines):
            if "epochs completed in" in line:
                match = re.search(r"(\d+)\s+epochs completed in\s+([\d.]+)\s+hours\s*\(([\d.]+)\s+minutes\)", line)
                if match:
                    epochs = match.group(1)
                    h, m = float(match.group(2)), float(match.group(3))
                    time_taken = f"{h:.2f} hours" if h >= 1.0 else f"{m:.1f} minutes"
            # Look for "Weights copied to dataset folder:"
            if "Weights copied to dataset folder:" in line or "Weights copied to:" in line:
                # Next line should be Best:
                if i + 1 < len(lines) and "Best:" in lines[i + 1]:
                    best_weights = lines[i + 1].split("Best:")[-1].strip()
                # Line after that should be Last:
                if i + 2 < len(lines) and "Last:" in lines[i + 2]:
                    last_weights = lines[i + 2].split("Last:")[-1].strip()

        epochs = epochs or "?"
        time_taken = time_taken or "?"
        last_weights = last_weights or "?"
        best_weights = best_weights or "?"
        # --- Only extract the last metrics block ---
        # Pattern for a metrics row (starts with all or class name, then numbers)
        metrics_pattern = re.compile(r"^\s*([\w\s-]+)\s+\d+\s+\d+\s+[\d.eE+-]+")
        metrics_block = []
        collecting = False
        for line in reversed(lines):
            if metrics_pattern.match(line):
                metrics_block.insert(0, line)
                collecting = True
            elif collecting:
                break

        # Parse metrics rows into columns
        class_metrics = []
        for line in metrics_block:
            parts = line.split()
            # The last 10 columns are always the metrics (numbers)
            if len(parts) > 10:
                class_name = " ".join(parts[:-10])
                metrics = parts[-10:]
                row = [class_name] + metrics
                class_metrics.append(row)

        # --- GUI Layout ---
        summary = QtWidgets.QGroupBox("Summary")
        summary_layout = QtWidgets.QFormLayout()
        summary_layout.addRow("Epochs:", QtWidgets.QLabel(str(epochs)))
        summary_layout.addRow("Time Taken:", QtWidgets.QLabel(str(time_taken)))
        summary_layout.addRow("Best Weights:", QtWidgets.QLabel(str(best_weights)))
        summary_layout.addRow("Last Weights:", QtWidgets.QLabel(str(last_weights)))
        #summary_layout.addRow("Results Directory:", QtWidgets.QLabel(str(results_dir)))
        summary.setLayout(summary_layout)
        layout.addWidget(summary)

        # Metrics Table
        self.table = QtWidgets.QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels([
            "Class", "Images", "Instances",
            "Box(P)", "Box(R)", "Box mAP50", "B mAP50-95",
            "Mask(P)", "Mask(R)", "Mask mAP50", "M mAP50-95"
        ])

        self.table.setRowCount(len(class_metrics))
        self.table.setMinimumHeight(250)
        self.table.verticalHeader().setVisible(False)
        for row, metrics in enumerate(class_metrics):
            row_data = metrics[:11] + [""] * (11 - len(metrics))
            for col, val in enumerate(row_data):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(val))
        self.table.resizeColumnsToContents()
        layout.addWidget(QtWidgets.QLabel("Final Per-Class and Overall Metrics:"))
        layout.addWidget(self.table)

        chart_layout = QtWidgets.QHBoxLayout()
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItems(["Mask mAP50", "Mask mAP50-95"])
        chart_layout.addWidget(QtWidgets.QLabel("Show:"))
        chart_layout.addWidget(self.metric_combo)
        layout.addLayout(chart_layout)

        self.class_chart = QtChart.QChartView()
        self.class_chart.setMinimumHeight(250)
        layout.addWidget(self.class_chart)

        # Close button
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)

        # Prepare data for chart
        self.class_names = [row[0] for row in class_metrics if row[0] != "all"]
        self.mask_map50 = [float(row[9]) if len(row) > 9 else 0 for row in class_metrics if row[0] != "all"]
        self.mask_map5095 = [float(row[10]) if len(row) > 10 else 0 for row in class_metrics if row[0] != "all"]

        self.metric_combo.currentIndexChanged.connect(self.update_class_chart)
        self.update_class_chart()

    def update_class_chart(self):
        chart = QtChart.QChart()
        chart.setTitle("Per-Class Mask mAP")
        bar_set = QtChart.QBarSet(self.metric_combo.currentText())
        if self.metric_combo.currentIndex() == 0:
            bar_set.append(self.mask_map50)
        else:
            bar_set.append(self.mask_map5095)
        series = QtChart.QBarSeries()
        series.append(bar_set)
        chart.addSeries(series)
        axis_x = QtChart.QBarCategoryAxis()
        axis_x.append(self.class_names)
        chart.addAxis(axis_x, QtCore.Qt.AlignBottom)
        series.attachAxis(axis_x)
        axis_y = QtChart.QValueAxis()
        axis_y.setRange(0, 1)
        chart.addAxis(axis_y, QtCore.Qt.AlignLeft)
        series.attachAxis(axis_y)
        self.class_chart.setChart(chart)

class TrainingDialog(QtWidgets.QDialog):
    cancelled_signal = pyqtSignal()
    def __init__(self, total_epochs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Training Progress")
        self.setWindowModality(QtCore.Qt.WindowModal)
        fit_dialog(self, 1000, 700)
        self.cancelled = False
        self.epoch_data = []
        self.current_epoch = 0
        self.total_epochs = total_epochs
        self.series = {}
        self._start_time = None

        self.setup_ui(total_epochs)

        self._clock_timer = QtCore.QTimer(self)
        self._clock_timer.setInterval(1000)
        self._clock_timer.timeout.connect(self._refresh_time_label)
        self._clock_timer.start()

    def setup_ui(self, total_epochs):
        layout = QtWidgets.QVBoxLayout()

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, total_epochs - 1)

        # Metrics table
        self.table = QtWidgets.QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "Epoch", "Box Loss", "Seg Loss", "DFL Loss", "Cls Loss",
            "Box mAP@0.5", "B mAP@0.5:0.95", "Mask mAP@0.5", "M mAP@0.5:0.95"
        ])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        for col in range(self.table.columnCount()):
            self.table.setColumnWidth(col, 108)

        self.table.setMinimumHeight(150)

        # Charts
        chart_layout = QtWidgets.QHBoxLayout()
        self.loss_chart = self.create_chart("Training Loss", ["box_loss", "seg_loss", "dfl_loss", "cls_loss"], total_epochs)
        self.map_chart = self.create_chart("Validation mAP", ["mask_map50", "mask_map5095"], total_epochs)
        chart_layout.addWidget(self.loss_chart)
        chart_layout.addWidget(self.map_chart)

        # Cancel button
        self.cancel_btn = QtWidgets.QPushButton("Cancel Training")
        self.cancel_btn.clicked.connect(self.cancel)

        # Time label
        self.time_label = QtWidgets.QLabel("Elapsed: --:--:--  |  ETA: --:--:--")
        self.time_label.setStyleSheet("color: #aaaaaa; font-style: italic;")

        # Assemble layout
        self.status_label = QtWidgets.QLabel("Initializing…")
        self.status_label.setStyleSheet("color: #88aaff; font-style: italic;")
        layout.addWidget(QtWidgets.QLabel("Epoch Progress:"))
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.time_label)
        layout.addWidget(self.status_label)
        layout.addWidget(QtWidgets.QLabel("Metrics:"))
        layout.addWidget(self.table)
        layout.addLayout(chart_layout)
        layout.addWidget(self.cancel_btn)
        self.setLayout(layout)

    def _fmt_seconds(self, secs):
        secs = int(max(0, secs))
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _refresh_time_label(self):
        if self._start_time is None:
            return
        import time
        elapsed = time.time() - self._start_time
        done = self.current_epoch + 1
        remaining_epochs = self.total_epochs - done
        if done > 0 and remaining_epochs > 0:
            secs_per_epoch = elapsed / done
            eta = secs_per_epoch * remaining_epochs
            eta_str = self._fmt_seconds(eta)
        else:
            eta_str = "--:--:--"
        self.time_label.setText(
            f"Elapsed: {self._fmt_seconds(elapsed)}  |  ETA: {eta_str}"
        )

    def set_status(self, msg):
        if hasattr(self, 'status_label'):
            self.status_label.setText(msg)

    def create_chart(self, title, series_names, total_epochs):
        chart = QtChart.QChart()
        chart.setTitle(title)
        chart.setMinimumHeight(200)
        chart.legend().setVisible(True)

        axis_x = QtChart.QValueAxis()
        axis_x.setTitleText("Epoch")
        axis_x.setRange(0, total_epochs - 1)
        # Show at most 10 ticks if epochs > 20, else show all
        max_ticks = 10
        if total_epochs > 20:
            axis_x.setTickCount(max_ticks)
        else:
            axis_x.setTickCount(total_epochs)
        axis_x.setLabelFormat("%d")

        chart.addAxis(axis_x, QtCore.Qt.AlignBottom)

        axis_y = QtChart.QValueAxis()
        axis_y.setTitleText("Value")
        chart.addAxis(axis_y, QtCore.Qt.AlignLeft)

        # Create series
        colors = [QtGui.QColor(c) for c in ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFFF00']]
        for i, name in enumerate(series_names):
            series = QtChart.QLineSeries()
            series.setName(name)
            series.setColor(colors[i % len(colors)])
            chart.addSeries(series)
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)
            self.series[name] = series

        chart_view = QtChart.QChartView(chart)
        chart_view.setRenderHint(QtGui.QPainter.Antialiasing)
        return chart_view
    
    def update_metrics(self, epoch, metrics):
        # Check if dialog still exists and is visible (prevents race condition)
        if not self or not self.isVisible():
            return

        import time
        if self._start_time is None:
            self._start_time = time.time()

        self.current_epoch = epoch
        self.progress_bar.setValue(epoch)
        row = epoch
        if row >= self.table.rowCount():
            self.table.insertRow(row)
        keys = [
            "epoch", "box_loss", "seg_loss", "dfl_loss", "cls_loss",
            "box_map50", "box_map5095", "mask_map50", "mask_map5095"
        ]
        for col, key in enumerate(keys):
            item = self.table.item(row, col)
            if item is None:
                item = QtWidgets.QTableWidgetItem("")
                self.table.setItem(row, col, item)
            value = metrics.get(key, "")
            if value != "" and value is not None:
                try:
                    if key == "epoch":
                        # Display epoch as integer
                        item.setText(str(int(value)))
                    else:
                        # Format numbers properly, including scientific notation
                        float_val = float(value)
                        if "map" in key.lower():
                            # For metrics, show up to 6 decimal places to handle very small values
                            if abs(float_val) < 0.0001:
                                item.setText(f"{float_val:.6f}")
                            else:
                                item.setText(f"{float_val:.4f}")
                        else:
                            # For losses
                            item.setText(f"{float_val:.5f}" if abs(float_val) < 10 else f"{float_val:.2f}")
                except (ValueError, TypeError):
                    item.setText(str(value))
            else:
                item.setText("")
        self.table.scrollToItem(self.table.item(epoch, 0))

        # --- Update charts ---
        for name in ["mask_map50", "mask_map5095", "box_loss", "seg_loss", "dfl_loss", "cls_loss"]:
            if name in self.series and name in metrics:
                try:
                    y = float(metrics[name])
                    self.series[name].append(epoch, y)
                except Exception:
                    pass

    def cancel(self):
        self.cancelled = True
        self._clock_timer.stop()
        self.cancelled_signal.emit()
        self.reject()

    def closeEvent(self, event):
        self._clock_timer.stop()
        if not self.cancelled:
            self.cancel()
        event.accept()

class RetrainWorker(QtCore.QObject):
    progress_signal = QtCore.pyqtSignal(int, dict)
    finished_signal = QtCore.pyqtSignal(str)
    status_signal = QtCore.pyqtSignal(str)
    error_signal = pyqtSignal(str)           # error message
    cancelled_signal = QtCore.pyqtSignal()          # training cancelled

    def __init__(self, data_yaml, params, parent=None):
        super().__init__(parent)
        self.data_yaml = data_yaml
        self.params = params
        self._process = None

    def stop(self):
        """Instantly kill the training subprocess."""
        if self._process and self._process.is_alive():
            self._process.kill()

    def run(self):
        import time
        import multiprocessing as mp
        from labelme.yolo_train_process import train_segmentation

        start_time = time.time()

        dataset_folder = os.path.dirname(self.data_yaml)
        is_valid, err = validate_yolo_dataset(dataset_folder)
        if not is_valid:
            self.error_signal.emit(f"Dataset Validation Error:\n\n{err}")
            return

        weights_path = self.params.get("weights_path", "")
        if not weights_path or not os.path.exists(weights_path):
            self.error_signal.emit(
                f"Weights file not found:\n{weights_path}\n\n"
                "Please select a valid .pt weights file for retraining."
            )
            return

        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        self._process = ctx.Process(
            target=train_segmentation,
            args=(self.data_yaml, self.params, q, weights_path),
            daemon=True,
        )
        self._process.start()

        last_metrics = {}
        error_msg = None
        best_weights_src = None
        last_weights_src = None

        while True:
            alive = self._process.is_alive()
            try:
                msg = q.get(timeout=0.3)
            except Exception:
                if not alive and q.empty():
                    exit_code = getattr(self._process, "exitcode", None)
                    if exit_code not in (0, None):
                        amp_hint = (
                            "\n\nThis usually means AMP (mixed precision) crashed your GPU driver.\n"
                            "Pascal GPUs (GTX 1060/1070/1080) have limited FP16 support.\n"
                            "AMP is automatically disabled for Pascal, but if the crash persists:\n"
                            "• Update your NVIDIA driver to version 520+\n"
                            "• Try a smaller batch size or image size"
                        ) if isinstance(exit_code, int) and exit_code < 0 else (
                            "\n\nPossible causes:\n"
                            "• CUDA / GPU driver error\n"
                            "• Out of GPU memory before first epoch\n"
                            "• Try a smaller batch size or image size"
                        )
                        error_msg = f"Retraining process crashed (exit code {exit_code}).{amp_hint}"
                    break
                continue

            kind = msg[0]
            if kind == 'progress':
                last_metrics = msg[2]
                self.progress_signal.emit(msg[1], msg[2])
            elif kind == 'status':
                self.status_signal.emit(msg[1])
            elif kind == 'done':
                _, best_weights_src, last_weights_src, tm = msg
                last_metrics['_trainer_metrics'] = tm
                break
            elif kind == 'error':
                error_msg = msg[1]
                break

        self._process.join(timeout=2)

        if error_msg:
            if "out of memory" in error_msg.lower():
                self.error_signal.emit(
                    "GPU Memory Error\n\nRetraining failed due to insufficient GPU memory.\n\n"
                    "Solutions:\n• Reduce batch size\n• Use a smaller image size\n"
                    f"• Close other GPU applications\n\nDetails: {error_msg}"
                )
            else:
                self.error_signal.emit(
                    f"Retraining Failed\n\n{error_msg}\n\nCheck the console for details."
                )
            return

        if best_weights_src is None:
            self.cancelled_signal.emit()
            self.finished_signal.emit("__cancelled__")
            return

        end_time = time.time()
        elapsed = end_time - start_time
        results_text = (
            f"{self.params['epochs']} epochs completed in "
            f"{elapsed/3600:.2f} hours ({(elapsed%3600)/60:.1f} minutes).\n\n"
        )
        try:
            tm = last_metrics.get('_trainer_metrics', {})
            bp, br = tm.get('metrics/precision(B)', 0), tm.get('metrics/recall(B)', 0)
            bm50, bm95 = tm.get('metrics/mAP50(B)', 0), tm.get('metrics/mAP50-95(B)', 0)
            mp_, mr = tm.get('metrics/precision(M)', 0), tm.get('metrics/recall(M)', 0)
            mm50, mm95 = tm.get('metrics/mAP50(M)', 0), tm.get('metrics/mAP50-95(M)', 0)
            for cls in tm.get('_per_class', []):
                results_text += (
                    f"{cls['name']:>20}      0      0 "
                    f"{cls['P']:>10.4f} {cls['R']:>10.4f} {cls['mAP50']:>10.4f} {cls['mAP']:>10.4f} "
                    f"{cls.get('mask_P',0):>10.4f} {cls.get('mask_R',0):>10.4f} "
                    f"{cls.get('mask_mAP50',0):>10.4f} {cls.get('mask_mAP',0):>10.4f}\n"
                )
            results_text += (
                f"{'all':>20}      0      0 "
                f"{bp:>10.4f} {br:>10.4f} {bm50:>10.4f} {bm95:>10.4f} "
                f"{mp_:>10.4f} {mr:>10.4f} {mm50:>10.4f} {mm95:>10.4f}\n\n"
            )
        except Exception as e:
            print(f"[WARNING] Could not format final metrics: {e}")

        if os.path.exists(best_weights_src) and os.path.exists(last_weights_src):
            try:
                best_dest, last_dest = save_weights_to_exp_folder(
                    dataset_folder, best_weights_src, last_weights_src
                )
                results_text += f"\n\nWeights copied to dataset folder:\nBest: {best_dest}\nLast: {last_dest}\n"
            except Exception as e:
                results_text += f"\n\nWarning: Could not copy weights: {e}\n"
        else:
            results_text += (
                f"\n\nWarning: Weight files not found.\n"
                f"Best: {best_weights_src}\nLast: {last_weights_src}\n"
            )

        self.finished_signal.emit(results_text)



class YoloRetrainDialog(QtWidgets.QDialog):
    def __init__(self, dataset_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Retrain Parameters")
        self.setFixedWidth(350)
        layout = QtWidgets.QFormLayout(self)

        # Weights file picker
        self.weights_edit = QtWidgets.QLineEdit()
        self.weights_btn = QtWidgets.QPushButton("Browse")

        # --- Set default weights path ---
        default_weights = self.get_latest_weights_path(dataset_path)
        if default_weights:
            self.weights_edit.setText(default_weights)

        def pick_weights():
            start_dir = os.path.dirname(self.weights_edit.text()) if self.weights_edit.text() else ""
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Weights File", start_dir, "PyTorch Weights (*.pt)")
            if path:
                self.weights_edit.setText(path)
        self.weights_btn.clicked.connect(pick_weights)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.weights_edit)
        hlayout.addWidget(self.weights_btn)
        layout.addRow("Weights file:", hlayout)

        # Epochs - flexible spinbox
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        layout.addRow("Epochs:", self.epochs_spin)

        # Batch size - dropdown with specific values
        self.batch_combo = QtWidgets.QComboBox()
        self.batch_combo.addItems(["4", "8", "16", "32", "64"])
        self.batch_combo.setCurrentText("8")
        layout.addRow("Batch size:", self.batch_combo)

        self.imgsz_combo = QtWidgets.QComboBox()
        self.imgsz_combo.addItems(["320", "640", "1280"])
        self.imgsz_combo.setCurrentText("640")
        layout.addRow("Image size:", self.imgsz_combo)

        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setToolTip("Lower learning rate (0.001) preserves learned features when retraining.")
        layout.addRow("Learning rate:", self.lr_spin)

        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["auto", "0", "1", "2", "3"])
        self.device_combo.setToolTip("Training device. 'auto' selects the first available GPU. CPU is not supported.")
        layout.addRow("Device:", self.device_combo)

        self.start_btn = QtWidgets.QPushButton("Start Retraining")
        self.start_btn.clicked.connect(self.accept)
        layout.addRow(self.start_btn)

    def get_params(self):
        return {
            "weights_path": self.weights_edit.text(),
            "epochs": self.epochs_spin.value(),
            "batch_size": int(self.batch_combo.currentText()),
            "imgsz": int(self.imgsz_combo.currentText()),
            "lr0": self.lr_spin.value(),
            "device": self.device_combo.currentText(),
        }
    def get_latest_weights_path(self, dataset_path):
        """Find the latest trained weights for retraining.
        
        Args:
            dataset_path: Path to the dataset folder
            
        Returns:
            str: Path to the latest best.pt weights file, or empty string if not found
        """
        if not dataset_path:
            return ""
        
        weights_root = os.path.join(dataset_path, "weight")
        if not os.path.exists(weights_root):
            print(f"[INFO] No weights folder found at: {weights_root}")
            return ""
        
        try:
            exp_folders = [
                d for d in os.listdir(weights_root) 
                if d.startswith("exp") and os.path.isdir(os.path.join(weights_root, d))
            ]
            
            if not exp_folders:
                print(f"[INFO] No experiment folders found in: {weights_root}")
                return ""
            
            # Sort by experiment number (exp1, exp2, etc.)
            def get_exp_num(folder_name):
                try:
                    return int(folder_name[3:]) if len(folder_name) > 3 else 0
                except ValueError:
                    return 0
                    
            exp_folders_sorted = sorted(exp_folders, key=get_exp_num, reverse=True)
            
            for exp_folder in exp_folders_sorted:
                candidate = os.path.join(weights_root, exp_folder, "best.pt")
                if os.path.exists(candidate):
                    print(f"[INFO] Found latest weights: {candidate}")
                    return candidate
            
            print(f"[WARNING] No best.pt files found in experiment folders")
            return ""
        except Exception as e:
            print(f"[ERROR] Failed to search for weights: {str(e)}")
            return ""