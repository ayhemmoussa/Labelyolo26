import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QObject, pyqtSignal

# Re-export dialog classes from yolo_train_obb to avoid duplication
from labelme.yolo_train_obb import (
    TrainingDialog_obb,
    TrainingResultsDialog_obb,
    validate_yolo_dataset,
    save_weights_to_exp_folder,
)


class RetrainWorker_obb(QObject):
    progress_signal = pyqtSignal(int, dict)
    finished_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    cancelled_signal = pyqtSignal()

    def __init__(self, data_yaml, params, parent=None):
        super().__init__(parent)
        self.data_yaml = data_yaml
        self.params = params
        self._process = None

    def stop(self):
        if self._process and self._process.is_alive():
            print("[DEBUG] Killing OBB retraining subprocess")
            self._process.kill()

    def run(self):
        import time
        import multiprocessing as mp
        from labelme.yolo_train_process import train_obb

        start_time = time.time()

        dataset_folder = os.path.dirname(self.data_yaml)
        is_valid, error_msg = validate_yolo_dataset(dataset_folder)
        if not is_valid:
            self.error_signal.emit(f"Dataset Validation Error:\n\n{error_msg}")
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
            target=train_obb,
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
            err = error_msg
            if "out of memory" in err.lower():
                self.error_signal.emit(
                    "GPU Memory Error\n\n"
                    "Retraining failed due to insufficient GPU memory.\n\n"
                    "Solutions:\n"
                    "• Reduce batch size in training settings\n"
                    "• Use a smaller image size\n"
                    "• Close other GPU applications\n"
                    "• Restart the application\n\n"
                    f"Technical details: {err}"
                )
            else:
                self.error_signal.emit(
                    f"Retraining Failed\n\n"
                    f"An error occurred during OBB retraining:\n\n"
                    f"{err}\n\n"
                    "Check the console output for more details."
                )
            return

        if best_weights_src is None:
            self.cancelled_signal.emit()
            self.finished_signal.emit("__cancelled__")
            return

        end_time = time.time()
        elapsed_seconds = end_time - start_time
        elapsed_hours = elapsed_seconds / 3600
        elapsed_minutes = (elapsed_seconds % 3600) / 60

        results_text = (
            f"{self.params['epochs']} epochs completed in "
            f"{elapsed_hours:.2f} hours ({elapsed_minutes:.1f} minutes).\n\n"
        )

        try:
            tm = last_metrics.get('_trainer_metrics', {})
            p   = tm.get('metrics/precision(B)', 0)
            r   = tm.get('metrics/recall(B)', 0)
            m50 = tm.get('metrics/mAP50(B)', 0)
            m95 = tm.get('metrics/mAP50-95(B)', 0)
            for cls in tm.get('_per_class', []):
                results_text += (
                    f"{cls['name']:>20}      0      0 "
                    f"{cls['P']:>10.4f} {cls['R']:>10.4f} {cls['mAP50']:>10.4f} {cls['mAP']:>10.4f}\n"
                )
            results_text += (
                f"{'all':>20}      0      0 "
                f"{p:>10.4f} {r:>10.4f} {m50:>10.4f} {m95:>10.4f}\n\n"
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
                f"\n\nWarning: Weight files not found at expected location.\n"
                f"Best: {best_weights_src}\nLast: {last_weights_src}\n"
            )

        self.finished_signal.emit(results_text)


class YoloRetrainDialog_obb(QtWidgets.QDialog):
    def __init__(self, dataset_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO OBB Retrain Parameters")
        self.setFixedWidth(350)
        layout = QtWidgets.QFormLayout(self)

        self.weights_edit = QtWidgets.QLineEdit()
        self.weights_btn = QtWidgets.QPushButton("Browse")

        default_weights = self.get_latest_weights_path(dataset_path)
        if default_weights:
            self.weights_edit.setText(default_weights)

        def pick_weights():
            start_dir = os.path.dirname(self.weights_edit.text()) if self.weights_edit.text() else ""
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select Weights File", start_dir, "PyTorch Weights (*.pt)"
            )
            if path:
                self.weights_edit.setText(path)

        self.weights_btn.clicked.connect(pick_weights)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.weights_edit)
        hlayout.addWidget(self.weights_btn)
        layout.addRow("Weights file:", hlayout)

        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        layout.addRow("Epochs:", self.epochs_spin)

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
        self.lr_spin.setToolTip("Initial learning rate. Use a lower value (0.001) for retraining to preserve learned features.")
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
        if not dataset_path:
            return ""
        weights_root = os.path.join(dataset_path, "weight")
        if not os.path.exists(weights_root):
            return ""
        try:
            exp_folders = [
                d for d in os.listdir(weights_root)
                if d.startswith("exp") and os.path.isdir(os.path.join(weights_root, d))
            ]
            if not exp_folders:
                return ""

            def get_exp_num(folder_name):
                try:
                    return int(folder_name[3:]) if len(folder_name) > 3 else 0
                except ValueError:
                    return 0

            exp_folders_sorted = sorted(exp_folders, key=get_exp_num, reverse=True)
            for exp_folder in exp_folders_sorted:
                candidate = os.path.join(weights_root, exp_folder, "best.pt")
                if os.path.exists(candidate):
                    return candidate
            return ""
        except Exception as e:
            print(f"[ERROR] Failed to search for weights: {str(e)}")
            return ""
