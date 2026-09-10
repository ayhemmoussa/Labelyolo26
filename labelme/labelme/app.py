# -*- coding: utf-8 -*-

import functools
import math
import os
import os.path as osp
import re
import webbrowser
import json
import hashlib
import sys

def get_python_executable():
    """Get the correct Python executable for running scripts"""
    # When running as a PyInstaller executable, sys.executable is the exe itself.
    # In that case we want to call the system Python that the user installed
    # (we assume the target machine will have Python on PATH). Allow override
    # via the PYTHON_EXECUTABLE environment variable.
    if getattr(sys, 'frozen', False):
        python_exe = os.environ.get('PYTHON_EXECUTABLE', 'python')
        print(f"[DEBUG] Frozen executable mode - using Python: {python_exe}")
        
        # Test if Python is available
        import subprocess
        try:
            result = subprocess.run([python_exe, '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"[DEBUG] Python test successful: {result.stdout.strip()}")
            else:
                print(f"[DEBUG] Python test failed with return code {result.returncode}")
                print(f"[DEBUG] Error: {result.stderr}")
        except Exception as e:
            print(f"[DEBUG] Python test exception: {e}")
        
        return python_exe
    else:
        print(f"[DEBUG] Development mode - using Python: {sys.executable}")
        return sys.executable

def get_resource_path(relative_path):
    """Get the correct path for resources in both development and packaged executable"""
    if hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller executable
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        # Running as normal Python script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, relative_path)

import imgviz
import natsort
import numpy as np
import torch
import sys
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from labelme import __appname__
#from labelme._automation import bbox_from_text
from labelme.config import get_config
from labelme.label_file import LabelFile
from labelme.label_file import LabelFileError
from labelme.shape import Shape
#from labelme.widgets import AiPromptWidget
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import Canvas
from labelme.widgets import FileDialogPreview
from labelme.widgets import LabelDialog
from labelme.widgets import LabelListWidget
from labelme.widgets import LabelListWidgetItem
from labelme.widgets import ToolBar
from labelme.widgets import UniqueLabelQListWidget
from labelme.widgets import ZoomWidget
from labelme.shape import Shape

from labelme.auto_label import auto_label_image
# magic_label is imported lazily inside MagicLabelWorker to avoid heavy startup deps
from labelme.data_split import TrainValSplitDialog
from labelme.stats_yolo import show_yolo_stats
from labelme.yolo_export import export_to_yolov5, set_dataset_path
from labelme.explore_yolo import ExploreYoloDialog
from labelme.yolo_train import YoloTrainDialog, TrainingDialog, TrainingResultsDialog, TrainingWorker, validate_yolo_dataset
from labelme.yolo_retrain import YoloRetrainDialog, TrainingDialog as RetrainTrainingDialog, TrainingResultsDialog as RetrainResultsDialog, RetrainWorker

from labelme.yolo_export_bbx import export_to_yolov5_bbx, set_dataset_path_bbx
from labelme.yolo_train_bbx import YoloTrainDialog_bbx, TrainingDialog_bbx,TrainingResultsDialog_bbx, TrainingWorker_bbx
from labelme.yolo_retrain_bbx import YoloRetrainDialog_bbx, TrainingDialog_bbx as RetrainTrainingDialog_bbx, TrainingResultsDialog_bbx as RetrainResultsDialog_bbx, RetrainWorker_bbx
from labelme.yolo_export_obb import export_to_yolo_obb, set_dataset_path_obb
from labelme.yolo_train_obb import YoloTrainDialog_obb, TrainingDialog_obb, TrainingResultsDialog_obb, TrainingWorker_obb
from labelme.yolo_retrain_obb import YoloRetrainDialog_obb, TrainingDialog_obb as RetrainTrainingDialog_obb, TrainingResultsDialog_obb as RetrainResultsDialog_obb, RetrainWorker_obb
from labelme.data_augmentation import DataAugmentationDialog
from . import utils

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - Zoom is too "steppy".


LABEL_COLORMAP = imgviz.label_colormap()

# Add this class to your app.py or a new file
class TaskTypeDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, default_task_type=None):
        super().__init__(parent)
        self.setWindowTitle("Select Task Type")
        utils.fit_dialog(self, 760, 370)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Select the annotation task type:"))

        btn_layout = QtWidgets.QHBoxLayout()

        from labelme.utils import newIcon

        _ACTIVE_BORDER = "QPushButton { border: 3px solid #0078d7; border-radius: 4px; }"

        # Segmentation button
        self.seg_btn = QtWidgets.QPushButton()
        self.seg_btn.setIcon(newIcon("segmentation2"))
        self.seg_btn.setIconSize(QtCore.QSize(200, 200))
        self.seg_btn.setFixedSize(220, 220)
        self.seg_btn.setToolTip("Segmentation — draw polygon masks")
        if default_task_type == "segmentation":
            self.seg_btn.setStyleSheet(_ACTIVE_BORDER)
        btn_layout.addWidget(self.seg_btn)

        # Bounding Box button
        self.bbox_btn = QtWidgets.QPushButton()
        self.bbox_btn.setIcon(newIcon("bbox2"))
        self.bbox_btn.setIconSize(QtCore.QSize(200, 200))
        self.bbox_btn.setFixedSize(220, 220)
        self.bbox_btn.setToolTip("Bounding Box — draw axis-aligned rectangles")
        if default_task_type == "bbox":
            self.bbox_btn.setStyleSheet(_ACTIVE_BORDER)
        btn_layout.addWidget(self.bbox_btn)

        # OBB button
        self.obb_btn = QtWidgets.QPushButton()
        self.obb_btn.setIcon(newIcon("obb"))
        self.obb_btn.setIconSize(QtCore.QSize(200, 200))
        self.obb_btn.setFixedSize(220, 220)
        self.obb_btn.setToolTip("OBB — draw oriented (rotated) bounding boxes")
        if default_task_type == "obb":
            self.obb_btn.setStyleSheet(_ACTIVE_BORDER)
        btn_layout.addWidget(self.obb_btn)

        layout.addLayout(btn_layout)

        # Labels below buttons
        label_layout = QtWidgets.QHBoxLayout()
        for text in ["Segmentation", "Bounding Box", "OBB"]:
            lbl = QtWidgets.QLabel(text)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            label_layout.addWidget(lbl)
        layout.addLayout(label_layout)

        # Show currently active mode if any
        if default_task_type:
            _names = {"segmentation": "Segmentation", "bbox": "Bounding Box", "obb": "OBB"}
            note = QtWidgets.QLabel(f"Current mode: <b>{_names.get(default_task_type, default_task_type)}</b> (highlighted)")
            note.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(note)

        self.task_type = None
        self.seg_btn.clicked.connect(self.select_seg)
        self.bbox_btn.clicked.connect(self.select_bbox)
        self.obb_btn.clicked.connect(self.select_obb)

    def select_seg(self):
        self.task_type = "segmentation"
        self.accept()

    def select_bbox(self):
        self.task_type = "bbox"
        self.accept()

    def select_obb(self):
        self.task_type = "obb"
        self.accept()

def _show_msgbox(icon, title, text, parent=None, min_width=520):
    """Show a QMessageBox with a guaranteed minimum width so the title bar never clips the content."""
    mb = QtWidgets.QMessageBox(icon, title, text, parent=parent)
    # Qt trick: inject a zero-height spacer that forces the desired minimum width
    spacer = QtWidgets.QSpacerItem(
        min_width, 0,
        QtWidgets.QSizePolicy.Minimum,
        QtWidgets.QSizePolicy.Expanding,
    )
    layout = mb.layout()
    layout.addItem(spacer, layout.rowCount(), 0, 1, layout.columnCount())
    mb.exec_()


def _is_yolov5_weights(path):
    """Return True if the .pt file is an old YOLOv5 checkpoint (not ultralytics).

    Ultralytics always stores 'train_args' as a key in the checkpoint dict.
    YOLOv5 never does.  We scan the entire pickle data to handle large
    checkpoints (e.g. OBB models) where 'train_args' may appear beyond
    the first few KB.
    """
    import zipfile
    try:
        with zipfile.ZipFile(path, "r") as zf:
            pkl_name = next((n for n in zf.namelist() if n.endswith("data.pkl")), None)
            if pkl_name is None:
                return False  # Unknown format — let YOLO() decide
            with zf.open(pkl_name) as pkl_file:
                data = pkl_file.read()
        return b"train_args" not in data
    except Exception:
        return False


class ModelLoadingWorker(QThread):
    """Worker thread for loading YOLO model"""
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object, str)  # (model, error_message)

    def __init__(self, weights_path):
        super().__init__()
        self.weights_path = weights_path

    def run(self):
        try:
            # Only check for old YOLOv5 format on existing local files (absolute paths).
            # Skip the check for pretrained model names like "yolo26n.pt" — those are
            # resolved and downloaded by ultralytics and are never YOLOv5 format.
            if os.path.isabs(self.weights_path) and _is_yolov5_weights(self.weights_path):
                self.finished_signal.emit(
                    None,
                    "Old YOLOv5 Model Detected\n\n"
                    "The weights file was trained with the old YOLOv5 framework,\n"
                    "which is no longer supported.\n\n"
                    "To use Auto Label you need to retrain your model:\n"
                    "  • Go to Project → Train YOLO\n"
                    "  • Train a new YOLO26 model on your dataset\n"
                    "  • The new model will be loaded automatically\n\n"
                    f"Old weights path:\n{self.weights_path}"
                )
                return

            self.status_signal.emit("Loading model weights...")
            from ultralytics import YOLO

            self.status_signal.emit("Initializing YOLO model...")
            model = YOLO(self.weights_path)

            self.status_signal.emit("Model loaded successfully!")
            self.finished_signal.emit(model, None)
        except Exception as e:
            self.finished_signal.emit(None, str(e))


class MagicLabelWorker(QThread):
    """Worker thread for Magic Label (Grounding DINO + SAM2)."""
    finished_signal = pyqtSignal(list)   # list of shape dicts
    error_signal    = pyqtSignal(str)    # error message
    status_signal   = pyqtSignal(str)    # progress status

    def __init__(self, image_path, text_prompt, box_threshold, text_threshold, use_sam):
        super().__init__()
        self.image_path     = image_path
        self.text_prompt    = text_prompt
        self.box_threshold  = box_threshold
        self.text_threshold = text_threshold
        self.use_sam        = use_sam
        self._stop          = False

    def cancel(self):
        self._stop = True

    def run(self):
        self._stop = False
        try:
            from labelme.magic_label import magic_label_image
            shapes = magic_label_image(
                self.image_path,
                self.text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                use_sam=self.use_sam,
                progress_callback=lambda msg: self.status_signal.emit(msg),
                stop_callback=lambda: self._stop,
            )
            if not self._stop:
                self.finished_signal.emit(shapes)
        except ImportError as e:
            self.error_signal.emit(
                f"Missing dependency:\n\n{str(e)}"
            )
        except Exception as e:
            import traceback
            self.error_signal.emit(
                f"Magic Label failed:\n\n{type(e).__name__}: {str(e)}\n\n"
                f"{traceback.format_exc()}"
            )


class _MagicProgressDialog(QtWidgets.QDialog):
    """Simple progress dialog for Magic Label (replaces QProgressDialog)."""
    canceled = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Magic Label")
        self.setWindowModality(Qt.WindowModal)
        utils.fit_dialog(self, 440, 140)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(14)

        self._lbl = QtWidgets.QLabel("Starting…")
        self._lbl.setWordWrap(True)
        self._lbl.setStyleSheet("font-size:13px;")
        lay.addWidget(self._lbl)

        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        lay.addWidget(self._bar)

        lay.addStretch()
        row = QtWidgets.QHBoxLayout()
        row.addStretch()
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.setFixedWidth(90)
        self._cancel_btn.clicked.connect(self._on_cancel)
        row.addWidget(self._cancel_btn)
        lay.addLayout(row)

    def _on_cancel(self):
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setText("Cancelling…")
        self.canceled.emit()

    def setLabelText(self, text: str):
        self._lbl.setText(text)


class ModelLoadingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loading YOLO Model")
        self.setModal(True)
        utils.fit_dialog(self, 550, 250)
        
        # Set dialog style
        self.setStyleSheet("""
            QDialog {
                background-color: #2a2a2a;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QtWidgets.QLabel("Loading YOLO Model")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #C1FFB5;")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Model info
        self.info_group = QtWidgets.QGroupBox("Model Information")
        info_layout = QtWidgets.QFormLayout()
        info_layout.setSpacing(8)
        
        self.path_label = QtWidgets.QLabel("Loading...")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #0066cc; font-size: 10px;")
        
        self.type_label = QtWidgets.QLabel("Loading...")
        self.type_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        
        self.classes_label = QtWidgets.QLabel("Loading...")
        self.classes_label.setWordWrap(True)
        self.classes_label.setStyleSheet("font-size: 10px;")
        
        info_layout.addRow("<b>Model Path:</b>", self.path_label)
        info_layout.addRow("<b>Task Type:</b>", self.type_label)
        info_layout.addRow("<b>Classes:</b>", self.classes_label)
        
        self.info_group.setLayout(info_layout)
        layout.addWidget(self.info_group)
        
        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #cccccc;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QtWidgets.QLabel("Initializing...")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-style: italic; font-size: 11px;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        self.worker = None
        self.model = None
        self.error_message = None
        
        # Center the dialog on screen
        self.center_on_screen()
    
    def center_on_screen(self):
        """Center the dialog on the screen"""
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        dialog_geometry = self.geometry()
        x = (screen.width() - dialog_geometry.width()) // 2
        y = (screen.height() - dialog_geometry.height()) // 2
        self.move(x, y)
        
    def set_model_info(self, path, task_type, classes):
        """Update the dialog with model information"""
        self.path_label.setText(path)
        
        if task_type == "segmentation":
            self.type_label.setText("Segmentation (Instance Segmentation)")
            self.type_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        elif task_type == "bbox":
            self.type_label.setText("Bounding Box (Object Detection)")
            self.type_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        elif task_type == "obb":
            self.type_label.setText("OBB (Oriented Bounding Box)")
            self.type_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        else:
            self.type_label.setText(task_type)
            self.type_label.setStyleSheet("font-weight: bold;")
        
        if classes:
            classes_str = ", ".join(classes)
            self.classes_label.setText(f"{len(classes)} classes: {classes_str}")
        else:
            self.classes_label.setText("No classes found")
    
    def set_status(self, status):
        """Update the status message"""
        self.status_label.setText(status)
    
    def load_model(self, weights_path):
        """Start loading the model in a worker thread"""
        self.worker = ModelLoadingWorker(weights_path)
        self.worker.status_signal.connect(self.set_status)
        self.worker.finished_signal.connect(self.on_model_loaded)
        self.worker.start()
    
    def on_model_loaded(self, model, error):
        """Called when model loading is complete"""
        self.model = model
        self.error_message = error
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        
        if error:
            self.set_status(f"Error: {error}")
        
        # Close dialog after a short delay to show final status
        QtCore.QTimer.singleShot(500, self.accept)

class DeleteLabelInUseDialog(QtWidgets.QDialog):
    def __init__(self, parent, label, other_labels, count):
        super().__init__(parent)
        self.setWindowTitle(f"Label '{label}' is in use")
        self.action = None
        self.target_label = None
        self.apply_all = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            f"The label '{label}' is used by {count} shape(s). Choose an action:"))
        self.reassign_radio = QtWidgets.QRadioButton("Reassign those shapes to:")
        self.delete_radio = QtWidgets.QRadioButton("Delete those shapes")
        self.cancel_radio = QtWidgets.QRadioButton("Cancel")
        self.cancel_radio.setChecked(True)
        self.apply_all_chk = QtWidgets.QCheckBox("Apply action to all images in dataset")
        self.apply_all_chk.setChecked(True)

        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(other_labels)
        self.combo.setEnabled(False)

        def update_state():
            self.combo.setEnabled(self.reassign_radio.isChecked())
        self.reassign_radio.toggled.connect(update_state)

        layout.addWidget(self.reassign_radio)
        h = QtWidgets.QHBoxLayout()
        h.addSpacing(30)
        h.addWidget(self.combo)
        layout.addLayout(h)
        layout.addWidget(self.delete_radio)
        layout.addWidget(self.cancel_radio)
        layout.addWidget(self.apply_all_chk)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(bb)
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)

    def _on_accept(self):
        if self.reassign_radio.isChecked():
            if self.combo.currentText().strip():
                self.action = "reassign"
                self.target_label = self.combo.currentText().strip()
                self.apply_all = self.apply_all_chk.isChecked()
                self.accept()
                return
            return
        if self.delete_radio.isChecked():
            self.action = "delete_shapes"
            self.apply_all = self.apply_all_chk.isChecked()
            self.accept()
            return
        self.reject()

class LabelListDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter Label List")
        layout = QtWidgets.QVBoxLayout(self)

        # Input row
        input_layout = QtWidgets.QHBoxLayout()
        self.line_edit = QtWidgets.QLineEdit()
        self.line_edit.setPlaceholderText("Enter label name")
        self.add_btn = QtWidgets.QPushButton("Add")
        input_layout.addWidget(self.line_edit)
        input_layout.addWidget(self.add_btn)
        layout.addLayout(input_layout)

        # List of labels
        self.list_widget = QtWidgets.QListWidget()
        layout.addWidget(self.list_widget)

        # OK/Cancel
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self.add_btn.clicked.connect(self.add_label)
        self.line_edit.returnPressed.connect(self.add_label)

    def add_label(self):
        text = self.line_edit.text().strip()
        if text and not self.label_exists(text):
            self.list_widget.addItem(text)
            self.line_edit.clear()

    def label_exists(self, text):
        for i in range(self.list_widget.count()):
            if self.list_widget.item(i).text() == text:
                return True
        return False

    def get_labels(self):
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())]

class _MaxSizeStack(QtWidgets.QStackedWidget):
    """QStackedWidget that always reports the maximum size hint across all pages,
    so the panel never shrinks when switching to a shorter page."""
    def sizeHint(self):
        sh = super().sizeHint()
        for i in range(self.count()):
            w = self.widget(i)
            if w:
                sh.setHeight(max(sh.height(), w.sizeHint().height()))
        return sh

    def minimumSizeHint(self):
        return self.sizeHint()


class FloatingAIPanel(QtWidgets.QFrame):
    """Draggable floating panel that unifies Auto Label and Magic Label controls."""

    # Signals emitted when the user clicks Run
    auto_label_requested        = QtCore.pyqtSignal(float, bool)                       # conf, append
    batch_auto_label_requested  = QtCore.pyqtSignal(float, bool)                       # conf, append (batch)
    magic_label_requested       = QtCore.pyqtSignal(str, str, float, float, bool, bool)
    batch_magic_label_requested = QtCore.pyqtSignal(str, str, float, float, bool, bool) # prompt,label_names,box,text,use_sam,append
    warmup_requested            = QtCore.pyqtSignal()

    _STYLE = """
        FloatingAIPanel {
            background: rgba(28, 28, 32, 225);
            border: 1px solid #555;
            border-radius: 10px;
        }
        QLabel  { color: #ccc; font-size: 11px; }
        QPushButton {
            background: #3a3a44; color: #ddd; border: 1px solid #555;
            border-radius: 4px; padding: 4px 8px; font-size: 11px;
        }
        QPushButton:hover  { background: #4a4a58; }
        QPushButton:checked { background: #5558a0; border-color: #888adf; color: #fff; }
        QPushButton#runBtn {
            background: #2d6a4f; color: #fff; font-weight: bold;
            border-color: #40916c; padding: 5px 10px;
        }
        QPushButton#runBtn:hover { background: #40916c; }
        QPushButton#batchBtn {
            background: #7a3e10; color: #fff; font-weight: bold;
            border-color: #c07840; padding: 5px 10px;
        }
        QPushButton#batchBtn:hover { background: #c07840; }
        QPushButton#closeBtn {
            background: transparent; border: none; color: #aaa;
            font-size: 16px; font-weight: bold; padding: 0;
        }
        QPushButton#closeBtn:hover { color: #fff; background: #c0392b; border-radius: 4px; }
        QSlider::groove:horizontal {
            height: 4px; background: #444; border-radius: 2px;
        }
        QSlider::handle:horizontal {
            width: 14px; height: 14px; margin: -5px 0;
            background: #888adf; border-radius: 7px;
        }
        QSlider::sub-page:horizontal { background: #5558a0; border-radius: 2px; }
        QLineEdit {
            background: #222; color: #eee; border: 1px solid #555;
            border-radius: 4px; padding: 3px 6px; font-size: 11px;
        }
        QCheckBox { color: #ccc; font-size: 11px; }
        QCheckBox::indicator { width: 13px; height: 13px; }
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet(self._STYLE)
        self.setFixedWidth(270)
        self._drag_pos = None
        self._setup_ui()
        self.hide()

    # ------------------------------------------------------------------ UI --

    def _setup_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(6)

        # ── Header ──────────────────────────────────────────────────────────
        hdr = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("AI Labeling")
        title.setStyleSheet("color:#aaa; font-size:10px; font-weight:bold; letter-spacing:1px;")
        close_btn = QtWidgets.QPushButton("✕")
        close_btn.setObjectName("closeBtn")
        close_btn.setFixedSize(26, 26)
        close_btn.clicked.connect(self.hide)
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(close_btn)
        root.addLayout(hdr)

        # thin divider
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet("color:#444;")
        root.addWidget(line)

        # ── Mode toggle ─────────────────────────────────────────────────────
        toggle = QtWidgets.QHBoxLayout()
        self._auto_btn  = QtWidgets.QPushButton("Auto Label")
        self._magic_btn = QtWidgets.QPushButton("Magic Label")
        for btn in (self._auto_btn, self._magic_btn):
            btn.setCheckable(True)
            btn.setFixedHeight(26)
            toggle.addWidget(btn)
        self._auto_btn.setChecked(True)
        self._auto_btn.clicked.connect(lambda: self.show_mode("auto"))
        self._magic_btn.clicked.connect(lambda: self.show_mode("magic"))
        root.addLayout(toggle)

        # ── Stacked pages ───────────────────────────────────────────────────
        self._stack = _MaxSizeStack()
        self._stack.addWidget(self._build_auto_page())
        self._stack.addWidget(self._build_magic_page())
        root.addWidget(self._stack)

    def _build_auto_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(4)

        self._conf_label = QtWidgets.QLabel("Confidence:  0.35")
        self._conf_slider = QtWidgets.QSlider(Qt.Horizontal)
        self._conf_slider.setRange(1, 100)
        self._conf_slider.setValue(35)
        self._conf_slider.valueChanged.connect(
            lambda v: self._conf_label.setText(f"Confidence:  {v/100:.2f}")
        )

        self._auto_append_check = QtWidgets.QCheckBox("Append")
        self._auto_append_check.setToolTip(
            "Add new detections on top of existing annotations.\n"
            "When unchecked, existing annotations are replaced."
        )

        self._nms_iou_label = QtWidgets.QLabel("Dedup IoU:  0.30")
        self._nms_iou_slider = QtWidgets.QSlider(Qt.Horizontal)
        self._nms_iou_slider.setRange(5, 95)
        self._nms_iou_slider.setValue(30)
        self._nms_iou_slider.setToolTip(
            "IoU threshold for duplicate removal in Append mode.\n"
            "Lower = more aggressive dedup (removes more overlapping boxes).\n"
            "Higher = only remove near-identical duplicates."
        )
        self._nms_iou_slider.valueChanged.connect(
            lambda v: self._nms_iou_label.setText(f"Dedup IoU:  {v/100:.2f}")
        )

        opts = QtWidgets.QHBoxLayout()
        opts.addStretch()
        opts.addWidget(self._auto_append_check)

        run_btn = QtWidgets.QPushButton("▶  Run Auto Label")
        run_btn.setObjectName("runBtn")
        run_btn.clicked.connect(
            lambda: self.auto_label_requested.emit(
                self._conf_slider.value() / 100.0,
                self._auto_append_check.isChecked(),
            )
        )

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet("color:#444; margin-top:2px; margin-bottom:2px;")

        batch_btn = QtWidgets.QPushButton("⚡  Label All Images in Folder")
        batch_btn.setObjectName("batchBtn")
        batch_btn.setToolTip(
            "Run auto-label on every image in the current folder\n"
            "and save annotations automatically.\n"
            "Append mode merges with existing annotations."
        )
        batch_btn.clicked.connect(
            lambda: self.batch_auto_label_requested.emit(
                self._conf_slider.value() / 100.0,
                self._auto_append_check.isChecked(),
            )
        )

        lay.addWidget(self._conf_label)
        lay.addWidget(self._conf_slider)
        lay.addWidget(self._nms_iou_label)
        lay.addWidget(self._nms_iou_slider)
        lay.addLayout(opts)
        lay.addSpacing(2)
        lay.addWidget(run_btn)
        lay.addWidget(sep)
        lay.addWidget(batch_btn)
        lay.addStretch()
        return page

    def _build_magic_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(4)

        lay.addWidget(QtWidgets.QLabel("Objects to detect:"))
        self._prompt_input = QtWidgets.QLineEdit()
        self._prompt_input.setPlaceholderText("defect in apple, crack on surface…")
        lay.addWidget(self._prompt_input)

        lay.addWidget(QtWidgets.QLabel("Label name(s) (optional):"))
        self._label_names_input = QtWidgets.QLineEdit()
        self._label_names_input.setPlaceholderText("defect, crack  (matches by position)")
        lay.addWidget(self._label_names_input)

        self._box_label = QtWidgets.QLabel("Box threshold:  0.30")
        self._box_slider = QtWidgets.QSlider(Qt.Horizontal)
        self._box_slider.setRange(1, 100)
        self._box_slider.setValue(30)
        self._box_slider.valueChanged.connect(
            lambda v: self._box_label.setText(f"Box threshold:  {v/100:.2f}")
        )

        self._text_label = QtWidgets.QLabel("Text threshold:  0.25")
        self._text_slider = QtWidgets.QSlider(Qt.Horizontal)
        self._text_slider.setRange(1, 100)
        self._text_slider.setValue(25)
        self._text_slider.valueChanged.connect(
            lambda v: self._text_label.setText(f"Text threshold:  {v/100:.2f}")
        )

        opts = QtWidgets.QHBoxLayout()
        self._sam2_check   = QtWidgets.QCheckBox("SAM2 masks")
        self._append_check = QtWidgets.QCheckBox("Append")
        self._sam2_check.setChecked(True)
        opts.addWidget(self._sam2_check)
        opts.addStretch()
        opts.addWidget(self._append_check)

        run_btn = QtWidgets.QPushButton("▶  Run Magic Label")
        run_btn.setObjectName("runBtn")
        run_btn.clicked.connect(self._emit_magic)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet("color:#444; margin-top:2px; margin-bottom:2px;")

        warmup_btn = QtWidgets.QPushButton("⚙  Pre-load AI Models")
        warmup_btn.setToolTip(
            "Load Grounding DINO (and SAM2 if enabled) into memory now\n"
            "so the first batch or single-image run starts instantly."
        )
        warmup_btn.clicked.connect(self._emit_warmup)

        magic_batch_btn = QtWidgets.QPushButton("⚡  Label All Images in Folder")
        magic_batch_btn.setObjectName("batchBtn")
        magic_batch_btn.setToolTip(
            "Run Magic Label on every image in the current folder.\n"
            "SAM2 masks are used when the SAM2 checkbox is enabled.\n"
            "Annotations are saved automatically."
        )
        magic_batch_btn.clicked.connect(self._emit_magic_batch)

        lay.addWidget(self._box_label)
        lay.addWidget(self._box_slider)
        lay.addWidget(self._text_label)
        lay.addWidget(self._text_slider)
        lay.addLayout(opts)
        lay.addSpacing(2)
        lay.addWidget(run_btn)
        lay.addWidget(sep)
        lay.addWidget(warmup_btn)
        lay.addWidget(magic_batch_btn)
        return page

    # ─────────────────────────────────────────────────────────── public API --

    def show_mode(self, mode: str):
        """Switch to the given mode ('auto' or 'magic') and make visible."""
        is_auto = (mode == "auto")
        self._auto_btn.setChecked(is_auto)
        self._magic_btn.setChecked(not is_auto)
        self._stack.setCurrentIndex(0 if is_auto else 1)
        if not self.isVisible():
            self._position_top_right()
        self.show()
        self.raise_()

    def sync_conf(self, value_0_to_1: float):
        """Keep the confidence slider in sync with the main threshold."""
        self._conf_slider.blockSignals(True)
        self._conf_slider.setValue(int(value_0_to_1 * 100))
        self._conf_label.setText(f"Confidence:  {value_0_to_1:.2f}")
        self._conf_slider.blockSignals(False)

    def set_prompt(self, text: str):
        self._prompt_input.setText(text)

    def configure_for_task(self, task_type: str):
        """Auto-configure SAM2 checkbox based on bbox vs segmentation mode."""
        is_seg = (task_type != "bbox")
        self._sam2_check.setChecked(is_seg)
        self._sam2_check.setEnabled(is_seg)

    def set_append_default(self, has_existing_shapes: bool):
        self._append_check.setChecked(has_existing_shapes)
        self._auto_append_check.setChecked(has_existing_shapes)

    # ────────────────────────────────────────────────────────────── private --

    def _emit_magic(self):
        self.magic_label_requested.emit(
            self._prompt_input.text(),
            self._label_names_input.text(),
            self._box_slider.value() / 100.0,
            self._text_slider.value() / 100.0,
            self._sam2_check.isChecked(),
            self._append_check.isChecked(),
        )

    def _emit_magic_batch(self):
        self.batch_magic_label_requested.emit(
            self._prompt_input.text(),
            self._label_names_input.text(),
            self._box_slider.value() / 100.0,
            self._text_slider.value() / 100.0,
            self._sam2_check.isChecked(),
            self._append_check.isChecked(),
        )

    def _emit_warmup(self):
        self.warmup_requested.emit()

    def nms_iou_threshold(self) -> float:
        """Return the current dedup IoU threshold from the slider (default 0.30)."""
        if hasattr(self, '_nms_iou_slider'):
            return self._nms_iou_slider.value() / 100.0
        return 0.30

    def _position_top_right(self):
        p = self.parent()
        if p:
            self.move(p.width() - self.width() - 16, 16)

    # ─────────────────────────────────────────────────────── drag to move --

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag_pos = ev.pos()

    def mouseMoveEvent(self, ev):
        if self._drag_pos and ev.buttons() == Qt.LeftButton:
            new_pos = self.pos() + ev.pos() - self._drag_pos
            # clamp inside parent
            p = self.parent()
            if p:
                new_pos.setX(max(0, min(new_pos.x(), p.width()  - self.width())))
                new_pos.setY(max(0, min(new_pos.y(), p.height() - self.height())))
            self.move(new_pos)

    def mouseReleaseEvent(self, _ev):
        self._drag_pos = None


class _ExportPreviewDialog(QtWidgets.QDialog):
    def __init__(self, image_path, shapes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Preview")
        utils.fit_dialog(self, 700, 520)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        info = QtWidgets.QLabel(f"Preview: {osp.basename(image_path)}  |  {len(shapes)} annotations")
        info.setStyleSheet("font-weight: bold; margin-bottom: 6px;")
        layout.addWidget(info)

        # Canvas showing the image with shapes
        self._canvas = QtWidgets.QLabel()
        self._canvas.setAlignment(Qt.AlignCenter)
        self._canvas.setMinimumSize(640, 400)
        self._canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self._canvas)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)

        # Load and render
        pix = QtGui.QPixmap(image_path)
        if not pix.isNull():
            orig_w = pix.width()
            orig_h = pix.height()
            pix = pix.scaled(640, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scale_x = pix.width() / orig_w if orig_w else 1.0
            scale_y = pix.height() / orig_h if orig_h else 1.0
            painter = QtGui.QPainter(pix)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            for shape in shapes:
                color = shape.line_color if shape.line_color else QtGui.QColor(0, 255, 0)
                pen = QtGui.QPen(color)
                pen.setWidthF(1.5)
                painter.setPen(pen)
                fill = QtGui.QColor(color)
                fill.setAlpha(40)
                painter.setBrush(fill)
                if shape.shape_type == "rotation":
                    raw_corners = shape._rotation_corners()
                    corners = QtGui.QPolygonF([
                        QtCore.QPointF(c.x() * scale_x, c.y() * scale_y)
                        for c in raw_corners
                    ])
                    painter.drawPolygon(corners)
                    pts = corners  # for label positioning below
                elif shape.shape_type == "rectangle" and len(shape.points) >= 2:
                    pts = [QtCore.QPointF(p.x() * scale_x, p.y() * scale_y) for p in shape.points]
                    rect = QtCore.QRectF(pts[0], pts[-1])
                    painter.drawRect(rect)
                else:
                    pts = [QtCore.QPointF(p.x() * scale_x, p.y() * scale_y) for p in shape.points]
                    if len(pts) >= 2:
                        poly = QtGui.QPolygonF(pts)
                        painter.drawPolygon(poly)
                # Draw label
                if pts and shape.label:
                    painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
                    font = QtGui.QFont()
                    font.setPointSize(7)
                    painter.setFont(font)
                    painter.drawText(QtCore.QPointF(pts[0].x() + 2, pts[0].y() - 2), shape.label)
            painter.end()
            self._canvas.setPixmap(pix)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        export_btn = QtWidgets.QPushButton("Export")
        export_btn.setDefault(True)
        export_btn.clicked.connect(self.accept)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(export_btn)
        layout.addLayout(btn_layout)


class MainWindow(QtWidgets.QMainWindow):
    def add_auto_label_slider(self):
        """Kept for compatibility — the floating AI panel now owns the confidence slider."""
        if not hasattr(self, "autoLabelThreshold"):
            self.autoLabelThreshold = 0.35


    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(
        self,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        if output is not None:
            logger.warning("argument output is deprecated, use output_file instead")
            if output_file is None:
                output_file = output

        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config

        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])  # type: ignore[assignment]
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])  # type: ignore[assignment]
        Shape.select_line_color = QtGui.QColor(  # type: ignore[assignment]
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(  # type: ignore[assignment]
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(  # type: ignore[assignment]
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(  # type: ignore[assignment]
            *self._config["shape"]["hvertex_fill_color"]
        )

        Shape.point_size = 8

        super(MainWindow, self).__init__()
        # Read saved task type before showing the dialog so we can skip it on
        # subsequent launches.  settings object is created again (idempotent) in
        # the settings-restore block below — QSettings is cheap to construct.
        _early_settings = QtCore.QSettings("labelme", "labelme")
        _saved_task_type = _early_settings.value("task_type", None)
        _valid_task_types = {"segmentation", "bbox", "obb"}
        # Always show the dialog so the user can change mode; pre-highlight the saved mode
        dialog = TaskTypeDialog(self, default_task_type=_saved_task_type)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.task_type = dialog.task_type
        elif _saved_task_type in _valid_task_types:
            self.task_type = _saved_task_type  # closed without choosing — keep previous
        else:
            sys.exit(0)  # No previous type and user cancelled

        # Set window title with mode
        _mode_names = {"segmentation": "Segmentation", "bbox": "Bounding Box", "obb": "OBB"}
        mode = _mode_names.get(self.task_type, self.task_type)
        self.setWindowTitle(f"{__appname__} - Mode: {mode}")

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False

        self._copied_shapes = None

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=False,
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self._config["label_flags"],
        )

        self.projectLabelList = QtWidgets.QListWidget()
        self.projectLabelList.setToolTip(
            self.tr("Labels defined in the selected dataset (.labels.json)")
        )
        self.projectLabelList.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.projectLabelList.setMinimumWidth(200)
        self.projectLabelList.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # type: ignore[attr-defined]

        self.projectLabelDock = QtWidgets.QDockWidget(self.tr("Project Labels"), self)
        self.projectLabelDock.setObjectName("ProjectLabels")
        self.projectLabelDock.setWidget(self.projectLabelList)
        self.projectLabelDock.setMinimumWidth(220)
        self.projectLabelDock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetClosable
        )

        self.labelList = LabelListWidget()
        self.lastOpenDir = None
        self.dataset_path = None

        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr("Flags"), self)
        self.flag_dock.setObjectName("Flags")
        self.flag_widget = QtWidgets.QListWidget()
        if config["flags"]:
            self.loadFlags({k: False for k in config["flags"]})
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.setDirty)
        
        # Initialize verified_empty state
        self.verified_empty = False

        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self._edit_label)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)
        
        # Create container for Polygon Labels with verified empty checkbox
        shape_container = QtWidgets.QWidget()
        shape_layout = QtWidgets.QVBoxLayout()
        shape_layout.setContentsMargins(0, 0, 0, 0)
        shape_layout.setSpacing(5)
        
        shape_layout.addWidget(self.labelList)
        
        # Add "Verified Empty" checkbox
        self.verified_empty_checkbox = QtWidgets.QCheckBox(self.tr("No Objects (Verified Empty)"))
        self.verified_empty_checkbox.setToolTip(
            self.tr("Check this to mark an image as verified to have no objects.\n"
                   "This will save the annotation and include the image in export for training.")
        )
        self.verified_empty_checkbox.stateChanged.connect(self.onVerifiedEmptyChanged)
        shape_layout.addWidget(self.verified_empty_checkbox)
        
        shape_container.setLayout(shape_layout)
        
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Polygon Labels"), self)
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(shape_container)
        self.shape_dock.setMinimumWidth(260)

        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(
            self.tr(
                "Select label to start annotating for it. " "Press 'Esc' to deselect."
            )
        )
        if self._config["labels"]:
            for label in self._config["labels"]:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
        
        # Search box for label list
        self.labelSearch = QtWidgets.QLineEdit()
        self.labelSearch.setPlaceholderText("🔍  Filter labels…")
        self.labelSearch.setClearButtonEnabled(True)
        self.labelSearch.textChanged.connect(self._filterLabelList)

        label_btn_layout = QtWidgets.QHBoxLayout()
        label_btn_layout.setContentsMargins(0, 0, 0, 0)

        self.add_label_btn = QtWidgets.QPushButton("+")
        self.del_label_btn = QtWidgets.QPushButton("-")
        self.up_label_btn = QtWidgets.QPushButton("↑")
        self.down_label_btn = QtWidgets.QPushButton("↓")
        self.up_label_btn.setFixedWidth(24)
        self.down_label_btn.setFixedWidth(24)
        self.add_label_btn.setFixedWidth(24)
        self.del_label_btn.setFixedWidth(24)

        label_btn_layout.addWidget(self.add_label_btn)
        label_btn_layout.addWidget(self.del_label_btn)
        label_btn_layout.addWidget(self.up_label_btn)
        label_btn_layout.addWidget(self.down_label_btn)
        label_btn_layout.addStretch()

        label_list_layout = QtWidgets.QVBoxLayout()
        label_list_layout.addWidget(self.labelSearch)
        label_list_layout.addLayout(label_btn_layout)
        label_list_layout.addWidget(self.uniqLabelList)

        label_list_widget = QtWidgets.QWidget()
        label_list_widget.setLayout(label_list_layout)

        self.add_label_btn.clicked.connect(self.addLabelToList)
        self.del_label_btn.clicked.connect(self.deleteLabelFromList)
        self.up_label_btn.clicked.connect(self.moveLabelUp)
        self.down_label_btn.clicked.connect(self.moveLabelDown)
        self.uniqLabelList.model().rowsMoved.connect(self.onLabelOrderChanged)
        self.uniqLabelList.itemSelectionChanged.connect(self.updateMoveButtons)
        self.updateMoveButtons()

        # Context menu for uniqLabelList (right-click on a label)
        self.uniqLabelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.uniqLabelList.customContextMenuRequested.connect(self._popUniqLabelListMenu)

        self.label_dock = QtWidgets.QDockWidget(self.tr("Label List"), self)
        self.label_dock.setObjectName("Label List")
        self.label_dock.setWidget(label_list_widget)
        self.label_dock.setMinimumWidth(260)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(self.fileSelectionChanged)
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("File List"), self)
        self.file_dock.setObjectName("Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        self.zoomWidget = ZoomWidget()
        self.setAcceptDrops(True)

        self.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
        )
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.mouseMoved.connect(
            lambda pos: self.status(f"Mouse is at: x={pos.x()}, y={pos.y()}")
        )

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollArea = scrollArea  # keep reference for floating panel
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),  # type: ignore[attr-defined]
            Qt.Horizontal: scrollArea.horizontalScrollBar(),  # type: ignore[attr-defined]
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas.shapesUpdated.connect(self._updateUndoIndicator)

        self.setCentralWidget(scrollArea)

        # Floating AI panel (parented to scroll area so it floats over the canvas)
        self._ai_panel = FloatingAIPanel(self.scrollArea)
        self._ai_panel.auto_label_requested.connect(self._run_auto_label_from_panel)
        self._ai_panel.batch_auto_label_requested.connect(self._run_batch_auto_label)
        self._ai_panel.batch_magic_label_requested.connect(self._run_batch_magic_label)
        self._ai_panel.magic_label_requested.connect(self._run_magic_label_from_panel)
        self._ai_panel.warmup_requested.connect(self.warmup_magic_label_models)

        # Periodic auto-save every 2 minutes when there are unsaved changes
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setInterval(120_000)
        self._autosave_timer.timeout.connect(self._autosave_tick)
        self._autosave_timer.start()

        # Quick review mode state
        self._review_queue: list = []
        self._review_index: int = 0
        self._in_review_mode: bool = False

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock"]:
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        # Add Project Labels dock first
        self.addDockWidget(Qt.RightDockWidgetArea, self.projectLabelDock)

        # Add the other docks
        self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)

        # Split Project Labels dock above the first dock (flag_dock)
        self.splitDockWidget(self.projectLabelDock, self.flag_dock, Qt.Vertical)

        # Actions
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config["shortcuts"]
        quit = action(
            self.tr("&Quit"),
            self.close,
            shortcuts["quit"],
            "quit",
            self.tr("Quit application"),
        )
        open_ = action(
            self.tr("&Open\n"),
            self.openFile,
            shortcuts["open"],
            "open",
            self.tr("Open image or label file"),
        )
        opendir = action(
            self.tr("Open Images"),
            self.openDirDialog,
            shortcuts["open_dir"],
            "open2",
            self.tr("Open Dir"),
        )
        openNextImg = action(
            self.tr("&Next Image"),
            self.openNextImg,
            shortcuts["open_next"],
            "next",
            self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        openPrevImg = action(
            self.tr("&Prev Image"),
            self.openPrevImg,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        copyToNext = action(
            self.tr("Copy to Next"),
            self.copyShapesToNext,
            None,
            "copy",
            self.tr("Copy all shapes from this image to the next image"),
        )
        save = action(
            self.tr("&Save\n"),
            self.saveFile,
            shortcuts["save"],
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            shortcuts["save_as"],
            "save-as",
            self.tr("Save labels to a different file"),
            enabled=False,
        )

        deleteFile = action(
            self.tr("&Delete Label-File"),
            self.deleteFile,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file"),
            enabled=False,
        )

        deleteImage = action(
            self.tr("Delete Image"),
            self.deleteImageFile,
            None,
            "delete_image",
            self.tr("Delete current image file from disk"),
            enabled=False,
        )

        changeOutputDir = action(
            self.tr("&Change Output Dir"),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts["save_to"],
            icon="open",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        saveAuto = action(
            text=self.tr("Save &Automatically"),
            slot=lambda x: self.actions.saveAuto.setChecked(x),  # type: ignore[attr-defined]
            icon="save",
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
        )
        saveAuto.setChecked(self._config["auto_save"])

        saveWithImageData = action(
            text=self.tr("Save With Image Data"),
            slot=self.enableSaveImageWithData,
            tip=self.tr("Save image data in label file"),
            checkable=True,
            checked=self._config["store_data"],
        )

        exploreDataset = utils.newAction(
            self,
            self.tr("Explore Dataset"),
            lambda: self.explore_yolo_dialog(),
            None,
            "explore",
            self.tr("Show all images in the dataset (train/val)"),
        )        

        trainYolo = utils.newAction(
            self,
            self.tr("Train YOLO"),
            lambda: self.start_yolo_training(),
            None,
            "yolo",
            self.tr("Train YOLO model on the current dataset"),
            #enabled=False,
        )

        exportYolo = utils.newAction(
            self,
            self.tr("Export YOLO"),
            self._exportYoloWithPreview,
            None,
            "yolo1",
            self.tr("Export labeled images and YOLO segmentation labels"),
            #enabled=False,
        )
        
        augmentData = utils.newAction(
            self,
            self.tr("Augment Dataset"),
            lambda: self.open_augmentation_dialog(),
            None,
            "augmentation",
            self.tr("Apply data augmentation to increase dataset size"),
        )
        
        showStats = utils.newAction(
            self,
            self.tr("Dataset Stats"),
            self.show_stats_smart,
            None,
            "stats",
            self.tr("Show statistics and charts for the selected dataset"),
        )

        autoLabel = utils.newAction(
            self,
            self.tr("Auto Label"),
            self.auto_label_image_slot,
            None,
            "ai",
            self.tr("Automatically label objects using deep learning"),
            enabled=False,
        )
        self.add_auto_label_slider()  # still creates the dock (hidden) for legacy sync

        magicLabel = utils.newAction(
            self,
            self.tr("Magic Label"),
            self.magic_label_slot,
            shortcuts.get("magic_label"),
            "magic_label",
            self.tr("Auto-label with Grounding DINO + SAM2 (no trained model needed)"),
            enabled=False,
        )

        clearMagicCache = utils.newAction(
            self,
            self.tr("Clear Magic Label Cache"),
            self.clear_magic_label_cache_slot,
            None,
            None,
            self.tr("Free GPU/CPU memory used by Magic Label models"),
        )

        warmupModels = utils.newAction(
            self,
            self.tr("Pre-load AI Models"),
            self.warmup_magic_label_models,
            None,
            None,
            self.tr("Load Grounding DINO / SAM2 into memory so first batch inference is instant"),
        )

        datasetStats = utils.newAction(
            self,
            self.tr("Annotation Statistics"),
            self.show_dataset_stats,
            None,
            None,
            self.tr("Show per-class annotation counts for the current project folder"),
        )

        verifyEmptyShortcut = utils.newAction(
            self,
            self.tr("Toggle Verified Empty (E)"),
            self.toggleVerifiedEmpty,
            "E",
            None,
            self.tr("Mark / unmark the current image as verified-empty (no objects)"),
            enabled=True,
        )

        setDatasetPath = utils.newAction(
            self,
            self.tr("Select Project"),
            lambda: self._set_dataset_path_wrapper(),
            None,
            "open1",
            self.tr("Set the root folder for YOLO dataset export and training"),
        )

        retrainYolo = utils.newAction(
            self,
            self.tr("Retrain YOLO"),
            lambda: self.start_yolo_retraining(),
            None,
            "reyolo",
            self.tr("Retrain YOLO model from existing weights"),
        )

        close = action(
            self.tr("&Close"),
            self.closeFile,
            shortcuts["close"],
            "close",
            self.tr("Close current file"),
        )

        toggle_keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            self.toggleKeepPrevMode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "keep previous annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])

        switchTaskType = action(
            self.tr("Switch Task Type"),
            self.switchTaskType,
            None,
            "swap",
            self.tr("Switch between Segmentation, Bounding Box, and OBB modes"),
        )

        createMode = action(
            self.tr("Create Polygons"),
            lambda: self.toggleDrawMode(False, createMode="polygon"),
            shortcuts["create_polygon"],
            "objects",
            self.tr("Start drawing polygons"),
            enabled=False if self.task_type == "bbox" else False,  # Disabled for bbox task
        )
        createRectangleMode = action(
            self.tr("Create Rectangle"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "objects",
            self.tr("Start drawing rectangles"),
            enabled=False if self.task_type == "segmentation" else False,  # Disabled for segmentation task
        )
        createCircleMode = action(
            self.tr("Create Circle"),
            lambda: self.toggleDrawMode(False, createMode="circle"),
            shortcuts["create_circle"],
            "objects",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        createLineMode = action(
            self.tr("Create Line"),
            lambda: self.toggleDrawMode(False, createMode="line"),
            shortcuts["create_line"],
            "objects",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        createPointMode = action(
            self.tr("Create Point"),
            lambda: self.toggleDrawMode(False, createMode="point"),
            shortcuts["create_point"],
            "objects",
            self.tr("Start drawing points"),
            enabled=False,
        )
        createLineStripMode = action(
            self.tr("Create LineStrip"),
            lambda: self.toggleDrawMode(False, createMode="linestrip"),
            shortcuts["create_linestrip"],
            "objects",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )

        editMode = action(
            self.tr("Edit Polygons"),
            self.setEditMode,
            shortcuts["edit_polygon"],
            "edit",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShape,
            shortcuts["duplicate_polygon"],
            "copy",
            self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        copy = action(
            self.tr("Copy Polygons"),
            self.copySelectedShape,
            shortcuts["copy_polygon"],
            "copy_clipboard",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Polygons"),
            self.pasteSelectedShape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied polygons"),
            enabled=False,
        )
        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
        )
        removePoint = action(
            text=self.tr("Remove Selected Point"),
            slot=self.removeSelectedPoint,
            shortcut=shortcuts["remove_selected_point"],
            icon="edit",
            tip=self.tr("Remove selected point from polygon"),
            enabled=False,
        )

        undo = action(
            self.tr("Undo\n"),
            self.undoShapeEdit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            shortcuts["hide_all_polygons"],
            icon="eye",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            shortcuts["show_all_polygons"],
            icon="eye",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )
        toggleAll = action(
            self.tr("&Toggle\nPolygons"),
            functools.partial(self.togglePolygons, None),
            shortcuts["toggle_all_polygons"],
            icon="eye",
            tip=self.tr("Toggle all polygons"),
            enabled=False,
        )

        help = action(
            self.tr("&Tutorial"),
            self.tutorial,
            icon="help",
            tip=self.tr("Show tutorial page"),
        )

        shortcuts_action = action(
            self.tr("&Keyboard Shortcuts"),
            self.showShortcutsDialog,
            "Ctrl+/",
            tip=self.tr("Show keyboard shortcuts"),
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoomBoxLayout = QtWidgets.QVBoxLayout()
        zoomLabel = QtWidgets.QLabel(self.tr("Zoom"))
        zoomLabel.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        zoomBoxLayout.addWidget(zoomLabel)
        zoomBoxLayout.addWidget(self.zoomWidget)
        zoom.setDefaultWidget(QtWidgets.QWidget())
        zoom.defaultWidget().setLayout(zoomBoxLayout)  # type: ignore[union-attr]
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmtShortcut(
                    "{},{}".format(shortcuts["zoom_in"], shortcuts["zoom_out"])
                ),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(
            self.tr("Zoom &In"),
            functools.partial(self.addZoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoomOut = action(
            self.tr("&Zoom Out"),
            functools.partial(self.addZoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoomOrg = action(
            self.tr("&Original size"),
            functools.partial(self.setZoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        keepPrevScale = action(
            self.tr("&Keep Previous Scale"),
            self.enableKeepPrevScale,
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightnessContrast = action(
            self.tr("&Brightness Contrast"),
            self.brightnessContrast,
            None,
            "color",
            self.tr("Adjust brightness and contrast"),
            enabled=False,
        )
        # Group zoom controls into a list for easier toggling.
        zoomActions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.zoomMode = self.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)  # type: ignore[attr-defined]
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(
            self.tr("&Edit Label"),
            self._edit_label,
            shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.setFillDrawing,
            None,
            "color",
            self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        if self._config["canvas"]["fill_drawing"]:
            fill_drawing.trigger()

        # Label list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)  # type: ignore[attr-defined]
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = utils.struct(  # type: ignore[assignment,method-assign]
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save,
            saveAs=saveAs,
            open=open_,
            close=close,
            deleteFile=deleteFile,
            deleteImage=deleteImage,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            switchTaskType=switchTaskType,
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            copy=copy,
            paste=paste,
            undoLastPoint=undoLastPoint,
            undo=undo,
            removePoint=removePoint,
            createMode=createMode,
            editMode=editMode,
            autoLabel=autoLabel,
            magicLabel=magicLabel,
            clearMagicCache=clearMagicCache,
            warmupModels=warmupModels,
            datasetStats=datasetStats,
            verifyEmptyShortcut=verifyEmptyShortcut,
            createRectangleMode=createRectangleMode,
            createCircleMode=createCircleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createLineStripMode=createLineStripMode,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            keepPrevScale=keepPrevScale,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            brightnessContrast=brightnessContrast,
            zoomActions=zoomActions,
            openNextImg=openNextImg,
            openPrevImg=openPrevImg,
            copyToNext=copyToNext,
            fileMenuActions=(open_, opendir, save, saveAs, close, quit),
            tool=(),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                edit,
                duplicate,
                copy,
                paste,
                delete,
                None,
                undo,
                undoLastPoint,
                None,
                removePoint,
                None,
                toggle_keep_prev_mode,
                None,
                switchTaskType,
                None,
                copyToNext,
            ),
            # menu shown at right click
            menu=(
                autoLabel,
                magicLabel,
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                editMode,
                edit,
                duplicate,
                copy,
                paste,
                delete,
                undo,
                undoLastPoint,
                removePoint,
            ),
            onLoadActive=(
                close,
                autoLabel,
                magicLabel,
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                editMode,
                brightnessContrast,
            ),
            onShapesPresent=(saveAs, hideAll, showAll, toggleAll),
        )

        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)  # type: ignore[attr-defined]
        # After actions are created in __init__ or after loading dataset:

        self.menus = utils.struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            labelList=labelMenu,
        )

        utils.addActions(
            self.menus.file,  # type: ignore[attr-defined]
            (
                setDatasetPath,      
                None,          
                #open_,
                openNextImg,
                openPrevImg,
                copyToNext,
                opendir,
                self.menus.recentFiles,  # type: ignore[attr-defined]
                save,
                #saveAs,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                close,
                deleteFile,
                deleteImage,
                None,
                exploreDataset,
                exportYolo,
                augmentData,
                showStats,
                trainYolo,
                retrainYolo,
                None,
                clearMagicCache,
                warmupModels,
                None,
                datasetStats,
                verifyEmptyShortcut,
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, (help, shortcuts_action))  # type: ignore[attr-defined]
        utils.addActions(
            self.menus.view,  # type: ignore[attr-defined]
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                fill_drawing,
                None,
                hideAll,
                showAll,
                toggleAll,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                keepPrevScale,
                None,
                fitWindow,
                fitWidth,
                None,
                brightnessContrast,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)  # type: ignore[attr-defined]

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.actions.menu)  # type: ignore[attr-defined]
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )


        self.tools = self.toolbar("Tools")
        self.tools.setStyleSheet("""
            QToolButton {
                min-width: 50px;
            }
            """)
        self.actions.tool = (  # type: ignore[attr-defined]
            setDatasetPath,
            None, 
            None,
            #open_,
            opendir,
            openPrevImg,
            openNextImg,
            copyToNext,
            save,
            deleteFile,
            deleteImage,
            None,
            None,
            autoLabel,
            magicLabel,
            createRectangleMode,
            createMode,
            editMode,
            duplicate,
            delete,
            undo,
            brightnessContrast,
            None,
            None,
            fitWindow,
            zoom,
            None,
            None,
            exploreDataset,
            showStats,
            exportYolo,
            augmentData,   
            trainYolo,
            retrainYolo,
            None,
        )

        self.statusBar().showMessage(str(self.tr("%s started.")) % __appname__)  # type: ignore[union-attr]
        self.statusBar().show()  # type: ignore[union-attr]

        self._undoLabel = QtWidgets.QLabel("  Undo: 0  ")
        self._undoLabel.setStyleSheet("color: #555; padding: 0 4px;")
        self._undoLabel.setToolTip("Number of undo steps available")
        self.statusBar().addPermanentWidget(self._undoLabel)  # type: ignore[union-attr]

        self._zoomLabel = QtWidgets.QLabel("  100%  ")
        self._zoomLabel.setStyleSheet("color: #aaa; padding: 0 4px;")
        self._zoomLabel.setToolTip("Current zoom level — click toolbar zoom box to change")
        self.statusBar().addPermanentWidget(self._zoomLabel)  # type: ignore[union-attr]

        if output_file is not None and self._config["auto_save"]:
            logger.warning(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.imagePath = None
        self.recentFiles = []  # type: ignore[var-annotated]
        self.maxRecent = 7
        self.otherData = None
        self.zoom_level = 100
        self.fit_window = False
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightnessContrast_values = {}
        self.scroll_values = {  # type: ignore[var-annotated]
            Qt.Horizontal: {},  # type: ignore[attr-defined]
            Qt.Vertical: {},  # type: ignore[attr-defined]
        }  # key=filename, value=scroll_value

        if filename is not None and osp.isdir(filename):
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename

        if config["file_search"]:
            self.fileSearch.setText(config["file_search"])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("labelme", "labelme")
        self.recentFiles = self.settings.value("recentFiles", []) or []
        self.dataset_path = self.settings.value("dataset_path", None)
        if self.dataset_path:
            if self.task_type == "segmentation":
                labels_json = os.path.join(self.dataset_path, ".labels.json")
                if os.path.exists(labels_json):
                    try:
                        with open(labels_json, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if data.get("type") is None or data.get("type") != "segmentation":
                            self.dataset_path = None
                            
                    except Exception as e:
                        self.dataset_path = None
                        QtWidgets.QMessageBox.warning(self, "Dataset Error", f"Failed to read .labels.json: {e}")
                        
            elif self.task_type == "bbox":
                labels_json = os.path.join(self.dataset_path, ".labels.json")
                if os.path.exists(labels_json):
                    try:
                        with open(labels_json, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if data.get("type") is None or data.get("type") != "bbox":
                            self.dataset_path = None

                    except Exception as e:
                        self.dataset_path = None
                        QtWidgets.QMessageBox.warning(self, "Dataset Error", f"Failed to read .labels.json: {e}")

            elif self.task_type == "obb":
                labels_json = os.path.join(self.dataset_path, ".labels.json")
                if os.path.exists(labels_json):
                    try:
                        with open(labels_json, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if data.get("type") is None or data.get("type") != "obb":
                            self.dataset_path = None
                    except Exception as e:
                        self.dataset_path = None
                        QtWidgets.QMessageBox.warning(self, "Dataset Error", f"Failed to read .labels.json: {e}")

            else:
                self.dataset_path = None
        
        size = self.settings.value("window/size", QtCore.QSize(600, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        if self.dataset_path:
            print(f"Restored dataset path: {self.dataset_path}")
            self.load_yolo_model(False)
        else:
            # No dataset saved — load pretrained YOLO26 so auto-label works immediately
            self._load_pretrained_yolo26()
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(state)
        self.updateProjectLabelList()

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queueEvent(functools.partial(self.loadFile, self.filename))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)
        self.zoomWidget.valueChanged.connect(lambda v: self._zoomLabel.setText(f"  {v}%  "))

        self.populateModeActions()

        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+/"), self).activated.connect(self.showShortcutsDialog)

        self.showMaximized()

        # self.firstStart = True
        # if self.firstStart:
        #    QWhatsThis.enterWhatsThisMode()

    def _set_dataset_path_wrapper(self):
       
        if self.task_type == "bbox":
            new_path = set_dataset_path_bbx(self, self.dataset_path or self.lastOpenDir, self.settings)
        elif self.task_type == "obb":
            new_path = set_dataset_path_obb(self, self.dataset_path or self.lastOpenDir, self.settings)
        else:
            new_path = set_dataset_path(self, self.dataset_path or self.lastOpenDir, self.settings)
        
        if new_path:
            # VALIDATE BEFORE SETTING - Check dataset type matches current mode
            labels_json = os.path.join(new_path, ".labels.json")
            if os.path.exists(labels_json):
                try:
                    with open(labels_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    dataset_type = data.get("type")
                    expected_type = self.task_type  # "segmentation", "bbox", or "obb"

                    _type_names = {"segmentation": "Segmentation", "bbox": "Bounding Box", "obb": "OBB"}
                    if dataset_type and dataset_type != expected_type:
                        # Show clear error with actionable guidance
                        mode_name = _type_names.get(self.task_type, self.task_type)
                        dataset_mode_name = _type_names.get(dataset_type, dataset_type)
                        
                        QtWidgets.QMessageBox.warning(
                            self, 
                            "Dataset Type Mismatch",
                            f"This dataset is for '{dataset_mode_name}' mode,\n"
                            f"but the application is currently in '{mode_name}' mode.\n\n"
                            f"Please:\n"
                            f"1. Switch task type (Edit → Switch Task Type), OR\n"
                            f"2. Select a dataset matching the current '{mode_name}' mode"
                        )
                        return  # Don't set dataset_path
                    
                except json.JSONDecodeError as e:
                    QtWidgets.QMessageBox.warning(
                        self, 
                        "Dataset Error", 
                        f"Failed to read .labels.json: Invalid JSON format\n\n{e}"
                    )
                    return
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self, 
                        "Dataset Error", 
                        f"Failed to read .labels.json: {e}"
                    )
                    return
            
            # Only set dataset_path if validation passed
            self.dataset_path = new_path
            self.settings.setValue("dataset_path", new_path)

            # If no .labels.json yet, prompt the user to create class labels
            labels_json_path = os.path.join(new_path, ".labels.json")
            if not os.path.exists(labels_json_path):
                label_dialog = LabelListDialog(self)
                label_dialog.setWindowTitle("Create Class Labels for New Dataset")
                if label_dialog.exec_() == QtWidgets.QDialog.Accepted:
                    label_names = label_dialog.get_labels()
                else:
                    label_names = []
                # Write .labels.json with labels + task type
                try:
                    with open(labels_json_path, "w", encoding="utf-8") as _f:
                        json.dump({"labels": label_names, "type": self.task_type}, _f, ensure_ascii=False, indent=2)
                except Exception as _e:
                    logger.error(f"Failed to write .labels.json: {_e}")
                # Populate uniqLabelList and labelDialog with the new classes
                self.uniqLabelList.clear()
                for _lbl in label_names:
                    _item = self.uniqLabelList.createItemFromLabel(_lbl)
                    self.uniqLabelList.addItem(_item)
                    _rgb = self._get_rgb_by_label(_lbl)
                    self.uniqLabelList.setItemLabel(_item, _lbl, _rgb)
                self.labelDialog.setLabels(label_names)

            # Update window title immediately
            dataset_name = os.path.basename(self.dataset_path)
            _mode_names = {"segmentation": "Segmentation", "bbox": "Bounding Box", "obb": "OBB"}
            mode = _mode_names.get(self.task_type, self.task_type)
            self.setWindowTitle(f"{__appname__} - Dataset: {dataset_name} [Not trained] - Mode: {mode}")

            # Update UI and load model
            self.updateProjectLabelList()
            self.load_yolo_model(True)

    # ------------------------------------------------------------------
    # Feature 2: Copy Shapes to Next Image
    # ------------------------------------------------------------------

    def copyShapesToNext(self):
        if not self.canvas.shapes:
            self.status(self.tr("No shapes to copy."))
            return
        self._shapes_to_paste = [s.copy() for s in self.canvas.shapes]
        self._pending_paste = True
        self.openNextImg()

    def _pastePendingShapes(self):
        if not getattr(self, "_pending_paste", False):
            return
        self._pending_paste = False
        shapes = getattr(self, "_shapes_to_paste", [])
        if not shapes:
            return
        for shape in shapes:
            self.canvas.shapes.append(shape)
            self.addLabel(shape)
        self.canvas.storeShapes()
        self.canvas.update()
        self.setDirty()
        self.status(self.tr(f"Pasted {len(shapes)} shapes from previous image."))

    # ------------------------------------------------------------------
    # Feature 3: Bulk Label Rename
    # ------------------------------------------------------------------

    def _popUniqLabelListMenu(self, point):
        menu = QtWidgets.QMenu(self)
        rename_action = menu.addAction(self.tr("Rename label everywhere\u2026"))
        rename_action.triggered.connect(self.bulkRenameLabel)
        menu.exec_(self.uniqLabelList.mapToGlobal(point))

    def bulkRenameLabel(self):
        items = self.uniqLabelList.selectedItems()
        if not items:
            return
        old_label = items[0].data(Qt.UserRole)
        if old_label is None:
            return

        new_label, ok = QtWidgets.QInputDialog.getText(
            self,
            self.tr("Rename Label"),
            self.tr(f'Rename "{old_label}" to:'),
            text=old_label,
        )
        if not ok or not new_label.strip() or new_label == old_label:
            return
        new_label = new_label.strip()

        similar = self._find_similar_label(new_label)
        if similar and similar != old_label:
            reply = QtWidgets.QMessageBox.question(
                self, "Similar Label Exists",
                f'"{new_label}" looks similar to existing label "{similar}".\n\nUse "{similar}" instead?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes,
            )
            if reply == QtWidgets.QMessageBox.Yes:
                new_label = similar

        if not self.lastOpenDir:
            self.status(self.tr("No directory open."))
            return

        import glob as _glob
        json_files = _glob.glob(osp.join(self.lastOpenDir, "*.json"))
        renamed_count = 0
        file_count = 0

        for json_path in json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                changed = False
                for shape in data.get("shapes", []):
                    if shape.get("label") == old_label:
                        shape["label"] = new_label
                        changed = True
                        renamed_count += 1
                if changed:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    file_count += 1
            except Exception:
                pass

        # Rename in current image (in memory)
        for item in self.labelList:
            shape = item.shape()
            if shape is not None and shape.label == old_label:
                shape.label = new_label
                disp = new_label if shape.group_id is None else f"{new_label} ({shape.group_id})"
                self.labelList.setItemLabel(item, disp, shape.fill_color.getRgb()[:3])

        # Update uniqLabelList
        found = self.uniqLabelList.findItemByLabel(old_label)
        if found:
            found.setData(Qt.UserRole, new_label)
            rgb = self._get_rgb_by_label(new_label)
            self.uniqLabelList.setItemLabel(found, new_label, rgb)

        if renamed_count > 0:
            self.setDirty()
            self._updateLabelCounts()

        self.status(self.tr(f'Renamed "{old_label}" \u2192 "{new_label}" in {renamed_count} shapes across {file_count} files.'))

    # ------------------------------------------------------------------
    # Feature 7: Export with Preview
    # ------------------------------------------------------------------

    def _exportYoloWithPreview(self):
        # Show export preview with current image if available
        if self.filename and self.canvas.shapes:
            dlg = _ExportPreviewDialog(self.filename, self.canvas.shapes, self)
            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                return
        # Run the actual export
        if self.task_type == "bbox":
            export_to_yolov5_bbx(self, self.dataset_path, self.lastOpenDir, self.uniqLabelList, TrainValSplitDialog)
        elif self.task_type == "obb":
            export_to_yolo_obb(self, self.dataset_path, self.lastOpenDir, self.uniqLabelList, TrainValSplitDialog)
        else:
            export_to_yolov5(self, self.dataset_path, self.lastOpenDir, self.uniqLabelList, TrainValSplitDialog)

    def open_augmentation_dialog(self):
        """Open the data augmentation dialog"""
        if not self.dataset_path:
            QtWidgets.QMessageBox.warning(
                self, 
                "Augmentation", 
                "Please select a dataset first using 'Select Project'."
            )
            return
        
        # Check if dataset has images and labels
        images_dir = os.path.join(self.dataset_path, "images")
        labels_dir = os.path.join(self.dataset_path, "labels")
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            QtWidgets.QMessageBox.warning(
                self,
                "Augmentation",
                "Dataset must have 'images' and 'labels' directories.\n"
                "Please export your dataset first."
            )
            return
        
        # Open augmentation dialog
        dialog = DataAugmentationDialog(self, self.dataset_path, self.task_type)
        dialog.exec_()

    def onLabelOrderChanged(self, *args):
        labels = [
            self.uniqLabelList.item(i).data(Qt.UserRole)
            for i in range(self.uniqLabelList.count())
        ]
        self.labelDialog.setLabels(labels)

    def updateMoveButtons(self):
        count = self.uniqLabelList.count()
        row = self.uniqLabelList.currentRow()
        disable_all = (count == 0 or row < 0)
        if disable_all:
            self.up_label_btn.setEnabled(False)
            self.down_label_btn.setEnabled(False)
            return
        self.up_label_btn.setEnabled(row > 0)
        self.down_label_btn.setEnabled(row < count - 1)

    def addLabelToList(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Add Label", "Label name:")
        if ok and text.strip():
            label = text.strip()
            similar = self._find_similar_label(label)
            if similar:
                reply = QtWidgets.QMessageBox.question(
                    self, "Similar Label Exists",
                    f'"{label}" looks similar to existing label "{similar}".\n\nUse "{similar}" instead?',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes,
                )
                if reply == QtWidgets.QMessageBox.Yes:
                    label = similar
            if not self.uniqLabelList.findItemByLabel(label):
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
                self.labelDialog.setLabels([self.uniqLabelList.item(i).data(Qt.UserRole) for i in range(self.uniqLabelList.count())])
                self.uniqLabelList.setCurrentItem(item)
        self.updateMoveButtons()
        if self.lastOpenDir:
            self.save_label_list_json(self.lastOpenDir)

    def deleteLabelFromList(self):
        items = self.uniqLabelList.selectedItems()
        if not items:
            QtWidgets.QMessageBox.warning(self, "Delete Label", "No label selected.")
            return
        for list_item in items:
            label = list_item.data(Qt.UserRole)
            current_shapes = self._shapes_with_label(label)
            files_used, saved_shapes = self._scan_folder_label_usage(label)
            label_json = osp.splitext(self.imagePath or "")[0] + ".json" if self.imagePath else None
            if self.output_dir and label_json:
                label_json = osp.join(self.output_dir, osp.basename(label_json))
            current_file_saved = label_json and osp.exists(label_json)
            dialog_count = saved_shapes if current_file_saved else (saved_shapes + len(current_shapes))

            if not current_shapes and saved_shapes == 0:
                self._remove_label_item(label)
                continue

            other_labels = [
                self.uniqLabelList.item(i).data(Qt.UserRole)
                for i in range(self.uniqLabelList.count())
                if self.uniqLabelList.item(i).data(Qt.UserRole) != label
            ]

            dlg = DeleteLabelInUseDialog(self, label, other_labels, dialog_count)
            # Allow dialog even if other_labels empty (user can still choose delete)
            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                continue

            modified_total = 0
            if dlg.apply_all:
                self.saveAllImages()
                files_used, saved_shapes = self._scan_folder_label_usage(label)

            if dlg.action == "reassign":
                if dlg.target_label == label:
                    continue
                if not other_labels:
                    QtWidgets.QMessageBox.warning(self, "Reassign not possible", "No other labels to reassign to.")
                    continue
                if current_shapes:
                    self._relabel_shapes(label, dlg.target_label)
                if dlg.apply_all:
                    _, modified_shapes = self._batch_modify_jsons(label, new_label=dlg.target_label)
                    modified_total = modified_shapes
                    if modified_shapes > 0 and self.uniqLabelList.findItemByLabel(dlg.target_label) is None:
                        u_item = self.uniqLabelList.createItemFromLabel(dlg.target_label)
                        self.uniqLabelList.addItem(u_item)
                        rgb = self._get_rgb_by_label(dlg.target_label)
                        self.uniqLabelList.setItemLabel(u_item, dlg.target_label, rgb)
                    # Reload current image to reflect changes
                    if self.filename:
                        self.loadFile(self.filename)
            elif dlg.action == "delete_shapes":
                self._remove_shapes(current_shapes)
                if dlg.apply_all:
                    _, modified_shapes = self._batch_modify_jsons(label, delete=True)
                    modified_total = modified_shapes
                    # Reload current image to reflect changes
                    if self.filename:
                        self.loadFile(self.filename)

            _, saved_after = self._scan_folder_label_usage(label)
            still_used_current = any(s.label == label for s in self.canvas.shapes)
            if not still_used_current and saved_after == 0:
                self._remove_label_item(label)

        self.labelDialog.setLabels([
            self.uniqLabelList.item(i).data(Qt.UserRole)
            for i in range(self.uniqLabelList.count())
        ])
        if self.uniqLabelList.count():
            self.uniqLabelList.setCurrentRow(
                min(self.uniqLabelList.count() - 1, self.uniqLabelList.currentRow())
            )
        self.updateMoveButtons()
        self.setDirty()
        if self.lastOpenDir:
            self.save_label_list_json(self.lastOpenDir)

    def saveAllImages(self):
        """Force-save (or create) JSON for every image in imageList."""
        original = self.filename
        original_index = self.imageList.index(original) if original and original in self.imageList else None

        autosave_prev = self._config["auto_save"]
        self._config["auto_save"] = True  # force writes

        total = len(self.imageList)
        prog = QProgressDialog("Saving all annotations...", "Cancel", 0, total, self)
        prog.setWindowTitle("Save All")
        prog.setMinimumDuration(300)
        prog.setValue(0)
        QtWidgets.QApplication.processEvents()

        for idx, path in enumerate(list(self.imageList)):
            if prog.wasCanceled():
                break
            if not osp.exists(path):
                prog.setValue(idx + 1)
                continue
            if path != self.filename:
                if not self.mayContinue():
                    break
                self.loadFile(path)
            self.saveFile()
            prog.setValue(idx + 1)
            QtWidgets.QApplication.processEvents()

        prog.close()

        if original_index is not None and self.imageList:
            self.loadFile(self.imageList[original_index])

        self._config["auto_save"] = autosave_prev

    def save_label_list_json(self, dirpath):
        """Save the current label list to .labels.json in dirpath."""
        labels = [self.uniqLabelList.item(i).data(Qt.UserRole) for i in range(self.uniqLabelList.count())]
        label_file = os.path.join(dirpath, ".labels.json")
        try:
            with open(label_file, "w", encoding="utf-8") as f:
                json.dump({"labels": labels}, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved label list to {label_file}")
        except Exception as e:
            logger.error(f"Failed to save label list: {e}")

    def moveLabelUp(self):
        row = self.uniqLabelList.currentRow()
        if row > 0:
            item = self.uniqLabelList.takeItem(row)
            self.uniqLabelList.insertItem(row - 1, item)
            self.uniqLabelList.setCurrentRow(row - 1)
            label = item.data(Qt.UserRole)
            rgb = self._get_rgb_by_label(label)
            self.uniqLabelList.setItemLabel(item, label, rgb)
            self.labelDialog.setLabels([self.uniqLabelList.item(i).data(Qt.UserRole) for i in range(self.uniqLabelList.count())])
        self.updateMoveButtons()
        if self.lastOpenDir:
            self.save_label_list_json(self.lastOpenDir)

    def moveLabelDown(self):
        row = self.uniqLabelList.currentRow()
        if row != -1 and row < self.uniqLabelList.count() - 1:
            item = self.uniqLabelList.takeItem(row)
            self.uniqLabelList.insertItem(row + 1, item)
            self.uniqLabelList.setCurrentRow(row + 1)
            label = item.data(Qt.UserRole)
            rgb = self._get_rgb_by_label(label)
            self.uniqLabelList.setItemLabel(item, label, rgb)
            self.labelDialog.setLabels([self.uniqLabelList.item(i).data(Qt.UserRole) for i in range(self.uniqLabelList.count())])
        self.updateMoveButtons()
        if self.lastOpenDir:
            self.save_label_list_json(self.lastOpenDir)

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)  # type: ignore[union-attr]
        if actions:
            utils.addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName("%sToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)  # type: ignore[attr-defined]
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.TopToolBarArea, toolbar)  # type: ignore[attr-defined]
        return toolbar

    # Support Functions

    def noShapes(self):
        return not len(self.labelList)

    def populateModeActions(self):
        tool, menu = self.actions.tool, self.actions.menu  # type: ignore[attr-defined]
        self.tools.clear()
        utils.addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()  # type: ignore[attr-defined]
        actions = (
            self.actions.createMode,  # type: ignore[attr-defined]
            self.actions.createRectangleMode,  # type: ignore[attr-defined]
            self.actions.createCircleMode,  # type: ignore[attr-defined]
            self.actions.createLineMode,  # type: ignore[attr-defined]
            self.actions.createPointMode,  # type: ignore[attr-defined]
            self.actions.createLineStripMode,  # type: ignore[attr-defined]
            #self.actions.createAiPolygonMode,  # type: ignore[attr-defined]
            #self.actions.createAiMaskMode,  # type: ignore[attr-defined]
            self.actions.editMode,  # type: ignore[attr-defined]
        )
        utils.addActions(self.menus.edit, actions + self.actions.editMenu)  # type: ignore[attr-defined]

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)  # type: ignore[attr-defined]

        if self._config["auto_save"] or self.actions.saveAuto.isChecked():  # type: ignore[attr-defined]
            # Don't auto-save JSON in explore mode — user saves explicitly to YOLO .txt
            if not (hasattr(self, '_explore_image_list') and self._explore_image_list):
                label_file = osp.splitext(self.imagePath)[0] + ".json"  # type: ignore[arg-type]
                if self.output_dir:
                    label_file_without_path = osp.basename(label_file)
                    label_file = osp.join(self.output_dir, label_file_without_path)
                self.saveLabels(label_file)
                return
        self.dirty = True
        self.actions.save.setEnabled(True)  # type: ignore[attr-defined]
        self._update_window_title()

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)  # type: ignore[attr-defined]
        # Polygon only in segmentation mode
        if self.task_type == "segmentation":
            self.actions.createMode.setEnabled(True)  # type: ignore[attr-defined]
        # Rectangle in bbox and obb modes only
        if self.task_type in ("bbox", "obb"):
            self.actions.createRectangleMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createCircleMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createLineMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createPointMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createLineStripMode.setEnabled(True)  # type: ignore[attr-defined]
        #self.actions.createAiPolygonMode.setEnabled(True)  # type: ignore[attr-defined]
        #self.actions.createAiMaskMode.setEnabled(True)  # type: ignore[attr-defined]
        self._update_window_title()

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)  # type: ignore[attr-defined]
        else:
            self.actions.deleteFile.setEnabled(False)  # type: ignore[attr-defined]
        if self.filename and os.path.exists(self.filename):
            self.actions.deleteImage.setEnabled(True)  # type: ignore[attr-defined]
        else:
            self.actions.deleteImage.setEnabled(False)  # type: ignore[attr-defined]

    def _update_window_title(self):
        """Rebuild the window title preserving dataset + mode context."""
        _mode_names = {"segmentation": "Segmentation", "bbox": "Bounding Box", "obb": "OBB"}
        mode = _mode_names.get(self.task_type, self.task_type)
        if getattr(self, "dataset_path", None):
            dataset_name = os.path.basename(self.dataset_path)
            base = f"{__appname__} - Dataset: {dataset_name} - Mode: {mode}"
        else:
            base = f"{__appname__} - Mode: {mode}"
        if self.filename is not None:
            fname = os.path.basename(self.filename)
            dirty = "*" if self.dirty else ""
            self.setWindowTitle(f"{base} — {fname}{dirty}")
        else:
            self.setWindowTitle(base)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:  # type: ignore[attr-defined]
            z.setEnabled(value)
        for action in self.actions.onLoadActive:  # type: ignore[attr-defined]
            # Don't enable createMode (polygon) if task_type is bbox or obb
            if action == self.actions.createMode and self.task_type in ("bbox", "obb"):  # type: ignore[attr-defined]
                continue
            # Don't enable createRectangleMode if task_type is segmentation
            if action == self.actions.createRectangleMode and self.task_type == "segmentation":  # type: ignore[attr-defined]
                continue
            action.setEnabled(value)

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)  # type: ignore[union-attr]

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self.otherData = None
        self.verified_empty = False
        if hasattr(self, 'verified_empty_checkbox'):
            self.verified_empty_checkbox.blockSignals(True)
            self.verified_empty_checkbox.setChecked(False)
            self.verified_empty_checkbox.blockSignals(False)
        self.canvas.resetState()
        # Update auto-label availability (will be disabled since no image)
        self.updateAutoLabelAvailability()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    def updateAutoLabelAvailability(self):
        """Update auto-label action availability based on model and image state"""
        if hasattr(self, 'actions') and hasattr(self.actions, 'autoLabel'):
            is_available = (
                hasattr(self, 'yolo_model') and 
                self.yolo_model is not None and 
                self.imagePath is not None and 
                os.path.exists(str(self.imagePath))
            )
            self.actions.autoLabel.setEnabled(is_available)

    # Callbacks

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)  # type: ignore[attr-defined]
        self._updateUndoIndicator()
        self._updateLabelCounts()

    def tutorial(self):
        url = "https://github.com/labelmeai/labelme/tree/main/examples/tutorial"  # NOQA
        webbrowser.open(url)

    def showShortcutsDialog(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea, QWidget
        from PyQt5.QtCore import Qt

        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        utils.fit_dialog(dlg, 520, 480)

        outer = QVBoxLayout(dlg)
        outer.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Keyboard Shortcuts")
        title.setStyleSheet("font-size: 15px; font-weight: bold; margin-bottom: 8px;")
        outer.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        grid = QtWidgets.QGridLayout(content)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)

        SHORTCUTS = [
            ("File", None),
            ("Open Image", "Ctrl+O"),
            ("Open Folder", "Ctrl+U"),
            ("Save", "Ctrl+S"),
            ("Save As", "Ctrl+Shift+S"),
            ("Close", "Ctrl+W"),
            ("Quit", "Ctrl+Q"),
            ("Navigation", None),
            ("Next Image", "D"),
            ("Previous Image", "A"),
            ("Labeling", None),
            ("Create Polygon", "Ctrl+N"),
            ("Create Rectangle", "Ctrl+R"),
            ("Edit Label", "Ctrl+E"),
            ("Delete Shape", "Delete"),
            ("Duplicate Shape", "Ctrl+D"),
            ("Undo", "Ctrl+Z"),
            ("Magic Label", "Ctrl+M"),
            ("Toggle Shapes", "T"),
            ("View", None),
            ("Zoom In", "Ctrl++"),
            ("Zoom Out", "Ctrl+-"),
            ("Fit Window", "Ctrl+F"),
            ("Fit Width", "Ctrl+Shift+F"),
            ("Reset Zoom", "Ctrl+0"),
            ("Help", None),
            ("Shortcuts", "Ctrl+/"),
        ]

        row = 0
        col = 0
        for action_name, key in SHORTCUTS:
            if key is None:
                if col == 2:
                    col = 0
                    row += 1
                hdr = QLabel(action_name)
                hdr.setStyleSheet("font-weight: bold; color: #888; font-size: 11px; margin-top: 8px;")
                grid.addWidget(hdr, row, col * 2, 1, 2)
                row += 1
            else:
                action_lbl = QLabel(action_name)
                action_lbl.setStyleSheet("padding: 2px 0;")
                key_lbl = QLabel(key)
                key_lbl.setStyleSheet(
                    "background: #2a2040; color: #ccccee; border-radius: 4px; "
                    "padding: 1px 6px; font-family: monospace; font-size: 11px;"
                )
                key_lbl.setAlignment(Qt.AlignCenter)
                grid.addWidget(action_lbl, row, col * 2)
                grid.addWidget(key_lbl, row, col * 2 + 1)
                col += 1
                if col == 2:
                    col = 0
                    row += 1

        dlg.exec_()

    def _filterLabelList(self, text):
        text = text.strip().lower()
        for i in range(self.uniqLabelList.count()):
            item = self.uniqLabelList.item(i)
            label = item.data(Qt.UserRole) or item.text()
            item.setHidden(bool(text) and text not in label.lower())

    def _updateUndoIndicator(self):
        n = max(0, len(self.canvas.shapesBackups) - 1)
        self._undoLabel.setText(f"  Undo: {n}  ")
        self._undoLabel.setStyleSheet(
            "color: #aaa; padding: 0 4px;" if n > 0 else "color: #555; padding: 0 4px;"
        )

    def _updateLabelCounts(self):
        """Update shape count displayed next to each label in the Label List dock."""
        counts = {}
        for item in self.labelList:
            shape = item.shape()
            if shape is not None:
                counts[shape.label] = counts.get(shape.label, 0) + 1

        for i in range(self.uniqLabelList.count()):
            list_item = self.uniqLabelList.item(i)
            label = list_item.data(Qt.UserRole)
            if label is None:
                continue
            rgb = self._get_rgb_by_label(label)
            count = counts.get(label, 0)
            self.uniqLabelList.setItemLabel(list_item, label, rgb, count=count)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)  # type: ignore[attr-defined]
        self.actions.undoLastPoint.setEnabled(drawing)  # type: ignore[attr-defined]
        self.actions.undo.setEnabled(not drawing)  # type: ignore[attr-defined]
        self.actions.delete.setEnabled(not drawing)  # type: ignore[attr-defined]

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        # OBB mode uses a rotatable rectangle instead of a plain rectangle
        if createMode == "rectangle" and getattr(self, "task_type", None) == "obb":
            createMode = "rotation_rect"

        draw_actions = {
            "polygon": self.actions.createMode,  # type: ignore[attr-defined]
            "rectangle": self.actions.createRectangleMode,  # type: ignore[attr-defined]
            "circle": self.actions.createCircleMode,  # type: ignore[attr-defined]
            "point": self.actions.createPointMode,  # type: ignore[attr-defined]
            "line": self.actions.createLineMode,  # type: ignore[attr-defined]
            "linestrip": self.actions.createLineStripMode,  # type: ignore[attr-defined]
        }

        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode

        if not edit and createMode == "rotation_rect":
            self.status(
                "OBB mode: left-click drag to draw box  |  right-click drag to rotate  |  Esc to cancel"
            )

        if edit:
            for draw_action in draw_actions.values():
                draw_action.setEnabled(True)
            # Apply task-specific disabling
            if hasattr(self, "task_type"):
                if self.task_type == "segmentation":
                    self.actions.createRectangleMode.setEnabled(False)
                elif self.task_type == "bbox":
                    self.actions.createMode.setEnabled(False)
                elif self.task_type == "obb":
                    self.actions.createMode.setEnabled(False)
        else:
            for draw_mode, draw_action in draw_actions.items():
                draw_action.setEnabled(createMode != draw_mode)
            # Apply task-specific disabling again
            if hasattr(self, "task_type"):
                if self.task_type == "segmentation":
                    self.actions.createRectangleMode.setEnabled(False)
                elif self.task_type == "bbox":
                    self.actions.createMode.setEnabled(False)
                elif self.task_type == "obb":
                    self.actions.createMode.setEnabled(False)

        self.actions.editMode.setEnabled(not edit)  # type: ignore[attr-defined]

    def setEditMode(self):
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles  # type: ignore[attr-defined]
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        
        # Filter by current dataset if one is set (show only files from current dataset)
        if self.dataset_path:
            filtered_files = []
            for f in files:
                # Check if file is within the current dataset path
                try:
                    if self.dataset_path in str(f) or os.path.commonpath([self.dataset_path, str(f)]) == self.dataset_path:
                        filtered_files.append(f)
                except (ValueError, TypeError):
                    # Different drives on Windows or invalid path comparison
                    pass
            files = filtered_files
        
        for i, f in enumerate(files):
            icon = utils.newIcon("labels")
            action = QtWidgets.QAction(
                icon, "&%d %s" % (i + 1, QtCore.QFileInfo(f).fileName()), self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))  # type: ignore[attr-defined]

    def validateLabel(self, label):
        if self._config["validate_label"] is None:
            return True
        if self._config["validate_label"] == "exact":
            # Allow new labels too (remove this branch to keep strict)
            return True
        return True

    def _edit_label(self, value=None):
        if not self.canvas.editing():
            return
        items = self.labelList.selectedItems()
        if not items:
            logger.warning("No label is selected, so cannot edit label.")
            return
        shape0 = items[0].shape()

        # --- Use the same dialog as newShape ---
        previous_text = self.labelDialog.edit.text()
        self.labelDialog.edit.hide()
        self.labelDialog.edit_group_id.hide()
        # Show label selection dialog with current label pre-selected
        text, flags, group_id, description = self.labelDialog.popUp(shape0.label)
        self.labelDialog.edit.show()
        self.labelDialog.edit_group_id.show()
        if not text:
            self.labelDialog.edit.setText(previous_text)
            return
        # ---------------------------------------

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        self.canvas.storeShapes()
        changed_any = False
        for it in items:
            shp = it.shape()
            if shp.label != text:
                shp.label = text
                self._update_shape_color(shp)
                changed_any = True
            if flags is not None and shp.flags != flags:
                shp.flags = flags
                changed_any = True
            if shp.group_id != group_id:
                shp.group_id = group_id
                changed_any = True
            if shp.description != description:
                shp.description = description
                changed_any = True
            # refresh item text
            disp = shp.label if shp.group_id is None else f"{shp.label} ({shp.group_id})"
            self.labelList.setItemLabel(it, disp, shp.fill_color.getRgb()[:3])

        if changed_any:
            # ensure label appears in unique list
            if self.uniqLabelList.findItemByLabel(text) is None:
                u_item = self.uniqLabelList.createItemFromLabel(text)
                self.uniqLabelList.addItem(u_item)
                rgb = self._get_rgb_by_label(text)
                self.uniqLabelList.setItemLabel(u_item, text, rgb)
            self.labelDialog.setLabels([
                self.uniqLabelList.item(i).data(Qt.UserRole)
                for i in range(self.uniqLabelList.count())
            ])
            self.canvas.update()
            self.setDirty()

    def _stable_label_index(self, label: str):
        h = hashlib.md5(label.encode("utf-8")).hexdigest()
        return int(h[:8], 16) % len(LABEL_COLORMAP)

    def _ensure_current_saved(self):
        if self.dirty and not (self._config["auto_save"] or self.actions.saveAuto.isChecked()):  # type: ignore[attr-defined]
            self.saveFile()

    def _shapes_with_label(self, label):
        return [s for s in self.canvas.shapes if s.label == label]

    def _relabel_shapes(self, old_label, new_label):
        changed = False
        for shape in self.canvas.shapes:
            if shape.label == old_label:
                shape.label = new_label
                self._update_shape_color(shape)
                changed = True
        if changed:
            # Just iterate; avoid .item() dependency
            for list_item in self.labelList:
                shp = list_item.shape()
                txt = shp.label if shp.group_id is None else f"{shp.label} ({shp.group_id})"
                self.labelList.setItemLabel(list_item, txt, shp.fill_color.getRgb()[:3])
            if self.uniqLabelList.findItemByLabel(new_label) is None:
                u_item = self.uniqLabelList.createItemFromLabel(new_label)
                self.uniqLabelList.addItem(u_item)
                rgb = self._get_rgb_by_label(new_label)
                self.uniqLabelList.setItemLabel(u_item, new_label, rgb)
            self.canvas.update()
            self.setDirty()
            self.labelDialog.setLabels([
                self.uniqLabelList.item(i).data(Qt.UserRole)
                for i in range(self.uniqLabelList.count())
            ])

    def _remove_shapes(self, shapes):
        if not shapes:
            return
        # Remove from label list (tolerant)
        self.remLabels(shapes, strict=False)
        # Remove from canvas
        for s in list(shapes):
            if s in self.canvas.shapes:
                self.canvas.shapes.remove(s)
        self.canvas.update()
        self.setDirty()

    def _remove_label_item(self, label):
        item = self.uniqLabelList.findItemByLabel(label)
        if item:
            row = self.uniqLabelList.row(item)
            self.uniqLabelList.takeItem(row)

    def _scan_folder_label_usage(self, label):
        """Return (file_count, shape_count_saved) across dataset/output_dir (saved JSON only)."""
        roots = []
        root_main = self.dataset_path or self.lastOpenDir
        if root_main and os.path.isdir(root_main):
            roots.append(root_main)
        if self.output_dir and self.output_dir not in roots and os.path.isdir(self.output_dir):
            roots.append(self.output_dir)
        if not roots:
            logger.debug(f"_scan_folder_label_usage: no roots (label={label})")
            return 0, 0
        seen = set()
        file_count = 0
        shape_total = 0
        for root in roots:
            for dirpath, _, files in os.walk(root):
                for f in files:
                    if not f.lower().endswith(".json"):
                        continue
                    fp = osp.join(dirpath, f)
                    if fp in seen:
                        continue
                    try:
                        with open(fp, "r", encoding="utf-8") as jf:
                            data = json.load(jf)
                    except Exception as e:
                        logger.debug(f"Skip JSON ({fp}): {e}")
                        continue
                    shapes = data.get("shapes", [])
                    c = sum(1 for s in shapes if s.get("label") == label)
                    if c:
                        file_count += 1
                        shape_total += c
                    seen.add(fp)
        return file_count, shape_total

    def _batch_modify_jsons(self, old_label, new_label=None, delete=False):
        """
        Relabel or delete shapes with old_label across all saved JSONs.
        Returns (files_modified, shapes_modified).
        """
        roots = []
        # Always include lastOpenDir (where images and JSONs are)
        if self.lastOpenDir and os.path.isdir(self.lastOpenDir):
            roots.append(self.lastOpenDir)
        # Optionally include output_dir if set and different
        if self.output_dir and self.output_dir not in roots and os.path.isdir(self.output_dir):
            roots.append(self.output_dir)
        # Optionally include dataset_path if set and different
        if self.dataset_path and self.dataset_path not in roots and os.path.isdir(self.dataset_path):
            roots.append(self.dataset_path)
        logger.info(f"Searching for JSONs in roots: {roots}")

        # Collect JSON file list first (avoid long blocking without UI)
        json_files = []
        for root in roots:
            for dirpath, _, files in os.walk(root):
                for f in files:
                    if f.lower().endswith(".json"):
                        fp = osp.join(dirpath, f)
                        logger.info(f"Found JSON file: {fp}")
                        json_files.append(fp)

        # Progress dialog
        prog = QProgressDialog(
            "Applying changes to dataset...", "Cancel", 0, len(json_files), self
        )
        prog.setWindowTitle("Batch Modify")
        prog.setMinimumDuration(400)
        prog.setValue(0)
        QtWidgets.QApplication.processEvents()

        files_modified = 0
        shapes_modified = 0
        for idx, fp in enumerate(json_files):
            if prog.wasCanceled():
                logger.info("Batch modify canceled by user.")
                break
            logger.info(f"Processing JSON: {fp}")
            try:
                with open(fp, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
            except Exception as e:
                logger.debug(f"Skip JSON ({fp}): {e}")
                prog.setValue(idx + 1)
                continue
            shapes = data.get("shapes", [])
            changed = False
            for s in list(shapes):
                if s.get("label") == old_label:
                    logger.info(f"Found label '{old_label}' in {fp}")
                    if delete:
                        shapes.remove(s)
                        changed = True
                        shapes_modified += 1
                    elif new_label:
                        s["label"] = new_label
                        changed = True
                        shapes_modified += 1
            if changed:
                data["shapes"] = shapes
                try:
                    with open(fp, "w", encoding="utf-8") as jf:
                        json.dump(data, jf, ensure_ascii=False, indent=2)
                    files_modified += 1
                except Exception as e:
                    logger.error(f"Failed writing {fp}: {e}")
            prog.setValue(idx + 1)
            QtWidgets.QApplication.processEvents()

        prog.close()
        logger.info(
            f"_batch_modify_jsons: old_label='{old_label}' "
            f"action={'delete' if delete else 'reassign'} "
            f"files_modified={files_modified} shapes_modified={shapes_modified}"
        )
        return files_modified, shapes_modified

    def fileSearchChanged(self):
        self.importDirImages(
            self.lastOpenDir,
            pattern=self.fileSearch.text(),
            load=False,
        )

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.mayContinue():
            return

        currIndex = self.imageList.index(str(item.text()))
        if currIndex < len(self.imageList):
            filename = self.imageList[currIndex]
            if filename:
                self.loadFile(filename)

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)  # type: ignore[attr-defined]
        self.actions.duplicate.setEnabled(n_selected)  # type: ignore[attr-defined]
        self.actions.copy.setEnabled(n_selected)  # type: ignore[attr-defined]
        self.actions.edit.setEnabled(n_selected)  # type: ignore[attr-defined]

    def addLabel(self, shape):
        if shape.group_id is None:
            text = shape.label
        else:
            text = "{} ({})".format(shape.label, shape.group_id)
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.actions.onShapesPresent:  # type: ignore[attr-defined]
            action.setEnabled(True)

        self._update_shape_color(shape)
        self.labelList.setItemLabel(label_list_item, text, shape.fill_color.getRgb()[:3])
        self._updateLabelCounts()

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(255, 255, 255)   # white fill, colored outline
        shape.hvertex_fill_color = QtGui.QColor(r, g, b)        # solid color on hover
        shape.fill_color = QtGui.QColor(r, g, b, 70)   # subtle fill
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 110)  # slightly stronger when selected

    def _find_similar_label(self, new_name):
        """Return the most similar existing label if one is found above threshold, else None.
        Checks case-insensitive exact match first, then fuzzy similarity >= 0.82.
        Does NOT return a match if new_name already exactly equals an existing label."""
        import difflib
        existing = [
            self.uniqLabelList.item(i).data(Qt.UserRole)
            for i in range(self.uniqLabelList.count())
            if self.uniqLabelList.item(i).data(Qt.UserRole)
        ]
        if not existing:
            return None
        # Exact same name — no suggestion needed
        if new_name in existing:
            return None
        # Case-insensitive exact match
        for lbl in existing:
            if lbl.lower() == new_name.lower():
                return lbl
        # Fuzzy match
        new_lower = new_name.lower()
        best_lbl, best_score = None, 0.0
        for lbl in existing:
            score = difflib.SequenceMatcher(None, new_lower, lbl.lower()).ratio()
            if score > best_score:
                best_score = score
                best_lbl = lbl
        if best_score >= 0.82:
            return best_lbl
        return None

    def _get_rgb_by_label(self, label):
        if self._config["shape_color"] == "auto":
            label_id = self._stable_label_index(label)
            return LABEL_COLORMAP[label_id]
        elif (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            return self._config["label_colors"][label]
        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]
        return (0, 255, 0)

    def remLabels(self, shapes, strict=True):
        """
        Remove label list items for given shapes.
        strict=False: silently skip shapes without list items.
        """
        for shape in shapes:
            try:
                item = self.labelList.findItemByShape(shape)
            except ValueError:
                if strict:
                    raise
                else:
                    continue
            if item:
                self.labelList.removeItem(item)
        self._updateLabelCounts()

    def load_label_list_json(self, dirpath):
        """Load label list from .labels.json in dirpath, or return None if not found."""
        label_file = os.path.join(dirpath, ".labels.json")
        if not os.path.exists(label_file):
            return None
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("labels", [])
        except Exception as e:
            logger.error(f"Failed to load label list: {e}")
            return None

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        if replace:
            self.labelList.clear()
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)
        self._updateLabelCounts()

    def loadLabels(self, shapes):
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags: dict = shape["flags"] or {}
            description = shape.get("description", "")
            group_id = shape["group_id"]
            other_data = shape["other_data"]

            if not points:
                # skip point-empty shape
                continue
            
            # Filter/convert shapes based on current task type
            if self.task_type == "obb":
                # OBB mode: pass rotation and polygon shapes through; skip others
                if shape_type not in ("rotation", "polygon", "rectangle"):
                    continue
            elif self.task_type == "bbox" and shape_type != "rectangle":
                # Convert polygon/other shapes to bounding rectangle instead of discarding
                if len(points) >= 2:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    points = [(min(xs), min(ys)), (max(xs), max(ys))]
                    shape_type = "rectangle"
                    print(f"[INFO] Converted '{label}' ({shape['shape_type']}) to rectangle in bbox mode")
                else:
                    continue
            elif self.task_type == "segmentation" and shape_type == "rectangle":
                print(f"[INFO] Skipping rectangle shape '{label}' in segmentation mode")
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                description=description,
                mask=shape["mask"],
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            # Set other_data first so the rotation angle is available when
            # computing the handle position below.
            shape.other_data = other_data

            # Rotation shapes: ensure 5-point format (4 actual corners + handle).
            # Old files stored 2 or 3 points (bbox_tl, bbox_br[, handle]) — convert them.
            if shape.shape_type == "rotation":
                if len(shape.points) < 5:
                    corners = shape._rotation_corners()  # handles len 2, 3, 4
                    if len(corners) == 4:
                        shape.points = list(corners)
                        shape.point_labels = [1, 1, 1, 1]
                        handle_pos = shape._rotation_handle_pos()
                        shape.points.append(handle_pos)
                        shape.point_labels.append(1)
                else:
                    # Recompute the handle so it's always correct after any edit
                    shape.points[4] = shape._rotation_handle_pos()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(flags)

            s.append(shape)
        self.loadShapes(s)

    def loadFlags(self, flags):
        self.flag_widget.clear()  # type: ignore[union-attr]
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # type: ignore[attr-defined]
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)  # type: ignore[attr-defined]
            self.flag_widget.addItem(item)  # type: ignore[union-attr]

    def saveLabels(self, filename):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    description=s.description,
                    shape_type=s.shape_type,
                    flags=s.flags,
                    mask=None
                    if s.mask is None
                    else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                )
            )
            if s.confidence is not None:
                data["confidence"] = round(float(s.confidence), 4)
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        for i in range(self.flag_widget.count()):  # type: ignore[union-attr]
            item = self.flag_widget.item(i)  # type: ignore[union-attr]
            key = item.text()  # type: ignore[union-attr]
            flag = item.checkState() == Qt.Checked  # type: ignore[attr-defined,union-attr]
            flags[key] = flag
        
        # Include verified_empty in otherData
        other_data = self.otherData.copy() if self.otherData else {}
        other_data['verified_empty'] = self.verified_empty
        
        try:
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))  # type: ignore[arg-type]
            imageData = self.imageData if self._config["store_data"] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=other_data,
                flags=flags,
            )
            self.labelFile = lf
            items = self.fileListWidget.findItems(self.imagePath, Qt.MatchExactly)  # type: ignore[arg-type,attr-defined]
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)  # type: ignore[attr-defined]
                self._set_file_item_color(items[0], True)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def duplicateSelectedShape(self):
        self.copySelectedShape()
        self.pasteSelectedShape()

    def pasteSelectedShape(self):
        self.loadShapes(self._copied_shapes, replace=False)
        self.setDirty()

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)  # type: ignore[attr-defined]

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)  # type: ignore[attr-defined]

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    # Callback functions:

    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)  # type: ignore[attr-defined]
        flags = {}
        group_id = None
        description = ""
        if self._config["display_label_popup"] or not text:
            previous_text = self.labelDialog.edit.text()
            # Hide the input box and group id field
            self.labelDialog.edit.hide()
            self.labelDialog.edit_group_id.hide()
            text, flags, group_id, description = self.labelDialog.popUp(text)
            # Show them again for next time
            self.labelDialog.edit.show()
            self.labelDialog.edit_group_id.show()
            if not text:
                self.labelDialog.edit.setText(previous_text)

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
        if text:
            similar = self._find_similar_label(text)
            if similar:
                reply = QtWidgets.QMessageBox.question(
                    self, "Similar Label Exists",
                    f'"{text}" looks similar to existing label "{similar}".\n\nUse "{similar}" instead?',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes,
                )
                if reply == QtWidgets.QMessageBox.Yes:
                    text = similar
            self.labelList.clearSelection()
            shape = self.canvas.setLastLabel(text, flags)
            shape.group_id = group_id
            shape.description = description
            self.addLabel(shape)
            self.actions.editMode.setEnabled(True)  # type: ignore[attr-defined]
            self.actions.undoLastPoint.setEnabled(False)  # type: ignore[attr-defined]
            self.actions.undo.setEnabled(True)  # type: ignore[attr-defined]
            self.setDirty()
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units  # type: ignore[union-attr]
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(int(value))  # type: ignore[union-attr]
        self.scroll_values[orientation][self.filename] = value

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)  # type: ignore[attr-defined]
        self.actions.fitWindow.setChecked(False)  # type: ignore[attr-defined]
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.setZoom(zoom_value)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,  # type: ignore[attr-defined]
                self.scrollBars[Qt.Horizontal].value() + x_shift,  # type: ignore[attr-defined,union-attr]
            )
            self.setScroll(
                Qt.Vertical,  # type: ignore[attr-defined]
                self.scrollBars[Qt.Vertical].value() + y_shift,  # type: ignore[attr-defined,union-attr]
            )

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)  # type: ignore[attr-defined]
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)  # type: ignore[attr-defined]
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def enableKeepPrevScale(self, enabled):
        self._config["keep_prev_scale"] = enabled
        self.actions.keepPrevScale.setChecked(enabled)  # type: ignore[attr-defined]

    def onNewBrightnessContrast(self, qimage):
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage), clear_shapes=False)

    def brightnessContrast(self, value):
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        result = dialog.exec_()

        if result == QtWidgets.QDialog.Accepted:
            brightness = dialog.slider_brightness.value()
            contrast = dialog.slider_contrast.value()
            self.brightnessContrast_values[self.filename] = (brightness, contrast)
        else:
            # User cancelled — clear stored values so next load uses originals
            self.brightnessContrast_values.pop(self.filename, None)

    def togglePolygons(self, value):
        flag = value
        for item in self.labelList:
            if value is None:
                flag = item.checkState() == Qt.Unchecked  # type: ignore[attr-defined]
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)  # type: ignore[attr-defined]

    def loadFile(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file
        if filename in self.imageList and (
            self.fileListWidget.currentRow() != self.imageList.index(filename)
        ):
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
            self.fileListWidget.repaint()
            return

        self.resetState()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False
        # assumes same name, but json extension
        self.status(str(self.tr("Loading %s...")) % osp.basename(str(filename)))
        label_file = osp.splitext(filename)[0] + ".json"
        if self.output_dir:
            label_file_without_path = osp.basename(label_file)
            label_file = osp.join(self.output_dir, label_file_without_path)
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p>"
                        "<p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.status(self.tr("Error reading %s") % label_file)
                return False
            self.imageData = self.labelFile.imageData
            self.imagePath = osp.join(
                osp.dirname(label_file),
                self.labelFile.imagePath,  # type: ignore[arg-type]
            )
            self.otherData = self.labelFile.otherData
        else:
            self.imageData = LabelFile.load_image_file(filename)
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None
        image = QtGui.QImage.fromData(self.imageData)  # type: ignore[arg-type]

        if image.isNull():
            formats = [
                "*.{}".format(fmt.data().decode())
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.status(self.tr("Error reading %s") % filename)
            return False
        self.image = image
        self.filename = filename
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        flags = {k: False for k in self._config["flags"] or []}
        if self.labelFile:
            self.loadLabels(self.labelFile.shapes)
            if self.labelFile.flags is not None:
                flags.update(self.labelFile.flags)
            # Load verified_empty flag
            if self.otherData and 'verified_empty' in self.otherData:
                self.verified_empty = self.otherData['verified_empty']
            else:
                self.verified_empty = False
        else:
            self.verified_empty = False
        
        # Update checkbox state
        if hasattr(self, 'verified_empty_checkbox'):
            self.verified_empty_checkbox.blockSignals(True)
            self.verified_empty_checkbox.setChecked(self.verified_empty)
            self.verified_empty_checkbox.blockSignals(False)
        
        self.loadFlags(flags)

        # In explore mode (navigating a YOLO dataset) there is no JSON label file;
        # load shapes from the corresponding .txt label file instead.
        if self._explore_image_list and not self.labelFile:
            self._load_explore_yolo_labels(str(filename))

        if self._config["keep_prev"] and self.noShapes():
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()
        self.canvas.setEnabled(True)
        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjustScale(initial=True)
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        # set brightness contrast values
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if self._config["keep_prev_brightness"] and self.recentFiles:
            brightness, _ = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if self._config["keep_prev_contrast"] and self.recentFiles:
            _, contrast = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        self.brightnessContrast_values[self.filename] = (brightness, contrast)
        if brightness is not None or contrast is not None:
            dialog.onNewValue(None)
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.canvas.setFocus()
        w = self.image.width()
        h = self.image.height()
        self.status(f"{osp.basename(str(filename))}  [{w} × {h}]")

        # --- Add this block here ---
        if hasattr(self, "_explore_image_list") and self._explore_image_list:
            dataset_path = self.dataset_path
            # Detect split from path
            if "/train/" in filename or "\\train\\" in filename:
                split = "train"
            elif "/val/" in filename or "\\val\\" in filename:
                split = "val"
            else:
                split = None
            self.actions.deleteFile.setEnabled(True)
            self.actions.deleteImage.setEnabled(True)
            # Update auto-label availability based on model and image state
            self.updateAutoLabelAvailability()

            if split:
                img_name = os.path.splitext(os.path.basename(filename))[0]
                label_dir = os.path.join(dataset_path, "labels", split)
                label_file = os.path.join(label_dir, img_name + ".txt")
                shapes = []
                if os.path.exists(label_file):
                    labels_json = os.path.join(dataset_path, ".labels.json")
                    class_list = []
                    if os.path.exists(labels_json):
                        import json
                        with open(labels_json, "r", encoding="utf-8") as f:
                            class_list = json.load(f).get("labels", [])
                    image = QtGui.QImage(filename)
                    w, h = image.width(), image.height()
                    with open(label_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            class_id = int(parts[0])
                            label = class_list[class_id] if class_id < len(class_list) else str(class_id)
                            coords = [float(x) for x in parts[1:]]
                            if len(coords) == 4:
                                cx, cy, bw, bh = coords
                                x1 = (cx - bw / 2) * w
                                y1 = (cy - bh / 2) * h
                                x2 = (cx + bw / 2) * w
                                y2 = (cy + bh / 2) * h
                                points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                                shape_type = "rectangle"
                            else:
                                points = []
                                for i in range(0, len(coords), 2):
                                    x = coords[i] * w
                                    y = coords[i+1] * h
                                    points.append((x, y))
                                shape_type = "polygon"
                            shapes.append({
                                "label": label,
                                "points": points,
                                "shape_type": shape_type,
                                "flags": {},
                                "group_id": None,
                                "description": "",
                                "other_data": {},
                                "mask": None,
                            })
                    if hasattr(self, "loadLabels") and shapes:
                        self.loadLabels(shapes)
        # --- End block ---

        self._pastePendingShapes()
        return True

    def resizeEvent(self, event):
        if (
            self.canvas
            and not self.image.isNull()
            and self.zoomMode != self.MANUAL_ZOOM
        ):
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        value = int(100 * value)
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e  # type: ignore[union-attr]
        h1 = self.centralWidget().height() - e  # type: ignore[union-attr]
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0  # type: ignore[union-attr]
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config["store_data"] = enabled
        self.actions.saveWithImageData.setChecked(enabled)  # type: ignore[attr-defined]

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        self.settings.setValue("task_type", self.task_type)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def dragEnterEvent(self, event):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self.mayContinue():
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def openNextImg(self, _value=False, load=True):
        # Use imageList2 if exploring, else imageList
        img_list = self.imageList2 if self.imageList2 else self.imageList
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(img_list) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = img_list[0]
        else:
            try:
                currIndex = img_list.index(self.filename)
            except ValueError:
                currIndex = 0
            if currIndex + 1 < len(img_list):
                filename = img_list[currIndex + 1]
            else:
                filename = img_list[-1]
        self.filename = filename

        if self.filename and load:
            self.loadFile(self.filename)

        self._config["keep_prev"] = keep_prev

    def openPrevImg(self, _value=False):
        img_list = self.imageList2 if self.imageList2 else self.imageList
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(img_list) <= 0:
            return

        if self.filename is None:
            return

        try:
            currIndex = img_list.index(self.filename)
        except ValueError:
            currIndex = 0
        if currIndex - 1 >= 0:
            filename = img_list[currIndex - 1]
            if filename:
                self.loadFile(filename)

        self._config["keep_prev"] = keep_prev

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + ["*%s" % LabelFile.suffix]
        )
        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filters)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        fileDialog.setWindowFilePath(path)
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            fileName = fileDialog.selectedFiles()[0]
            if fileName:
                self.loadFile(fileName)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(  # type: ignore[union-attr]
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()  # type: ignore[union-attr]

        current_filename = self.filename
        self.importDirImages(self.lastOpenDir, load=False)

        if current_filename in self.imageList:
            # retain currently selected file
            self.fileListWidget.setCurrentRow(self.imageList.index(current_filename))
            self.fileListWidget.repaint()

    def saveFile(self, _value=False):
        """Save label file with error handling"""
        try:
            # In explore mode, save directly back to the YOLO .txt label file
            if hasattr(self, '_explore_image_list') and self._explore_image_list:
                self._save_explore_yolo_label()
                return
            
            assert not self.image.isNull(), "cannot save empty image"
            
            if not self.imagePath:
                self.status("No image loaded to save label for.")
                QtWidgets.QMessageBox.warning(
                    self,
                    "Cannot Save",
                    "No image is currently loaded.\n\n"
                    "Please open an image first."
                )
                return
            
            # Verify image still exists
            if not os.path.exists(self.imagePath):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Image Not Found",
                    f"The image file no longer exists:\n{self.imagePath}\n\n"
                    f"Cannot save labels for a missing image."
                )
                return
            
            label_file = os.path.splitext(self.imagePath)[0] + ".json"
            
            if self.output_dir:
                # Verify output directory exists
                if not os.path.exists(self.output_dir):
                    try:
                        os.makedirs(self.output_dir, exist_ok=True)
                    except (OSError, PermissionError) as e:
                        QtWidgets.QMessageBox.critical(
                            self,
                            "Cannot Create Directory",
                            f"Failed to create output directory:\n{self.output_dir}\n\n"
                            f"Error: {str(e)}\n\n"
                            f"Please check permissions or select a different location."
                        )
                        return
                
                label_file = osp.join(self.output_dir, osp.basename(label_file))
            
            # Check write permissions
            label_dir = os.path.dirname(label_file)
            if not os.access(label_dir, os.W_OK):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Permission Denied",
                    f"No write permission for directory:\n{label_dir}\n\n"
                    f"Please check folder permissions."
                )
                return
            
            self._saveFile(label_file)
            
        except AssertionError as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot Save",
                f"Cannot save: {str(e)}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Save Error",
                f"An error occurred while saving:\n\n"
                f"{type(e).__name__}: {str(e)}\n\n"
                f"Your labels may not have been saved."
            )
            import traceback
            traceback.print_exc()

    def saveFileAs(self, _value=False):
        # Behave the same as saveFile: always save next to the image
        self.saveFile()

    """  def saveFileDialog(self):
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(self, caption, self.output_dir, filters)
        else:
            dlg = QtWidgets.QFileDialog(self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename  # type: ignore[assignment]
        return filename """

    def _save_explore_yolo_label(self):
        """Save current canvas shapes back to the YOLO .txt label file when in explore mode."""
        import json as _json
        filename = self.filename
        if not filename:
            return

        # Detect split from path
        if "/train/" in filename or "\\train\\" in filename:
            split = "train"
        elif "/val/" in filename or "\\val\\" in filename:
            split = "val"
        else:
            QtWidgets.QMessageBox.warning(
                self, "Save Error",
                "Cannot determine train/val split from image path.\n\nMake sure the image is inside a 'train' or 'val' folder."
            )
            return

        dataset_path = self.dataset_path
        img_name = os.path.splitext(os.path.basename(filename))[0]
        label_dir = os.path.join(dataset_path, "labels", split)
        label_file = os.path.join(label_dir, img_name + ".txt")

        # Load class list
        labels_json = os.path.join(dataset_path, ".labels.json")
        class_list = []
        labels_data = {}
        if os.path.exists(labels_json):
            with open(labels_json, "r", encoding="utf-8") as f:
                labels_data = _json.load(f)
                class_list = labels_data.get("labels", [])

        # Image dimensions
        w = self.image.width()
        h = self.image.height()
        if w == 0 or h == 0:
            QtWidgets.QMessageBox.warning(self, "Save Error", "Image dimensions are invalid.")
            return

        # Convert shapes to YOLO format lines
        lines = []
        new_labels_added = False
        for shape in self.canvas.shapes:
            label = shape.label
            shape_type = shape.shape_type
            points = shape.points  # list of QPointF

            if label in class_list:
                class_id = class_list.index(label)
            else:
                class_list.append(label)
                class_id = len(class_list) - 1
                new_labels_added = True

            if shape_type == "rectangle":
                xs = [p.x() for p in points]
                ys = [p.y() for p in points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                cx = (x_min + x_max) / 2 / w
                cy = (y_min + y_max) / 2 / h
                bw = (x_max - x_min) / w
                bh = (y_max - y_min) / h
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            elif shape_type == "rotation":
                # OBB: write 4 corner points as 8 normalized coordinates.
                # _rotation_corners() returns the 4 corners from either the stored
                # corners (new 5-point format) or computed from bbox+angle (old format).
                # Point[4] is the visual rotation handle — must NOT be exported.
                corners = shape._rotation_corners()
                if not corners:
                    continue
                coords = []
                for p in corners:
                    coords.append(f"{max(0.0, min(1.0, p.x() / w)):.6f}")
                    coords.append(f"{max(0.0, min(1.0, p.y() / h)):.6f}")
                lines.append(f"{class_id} {' '.join(coords)}")
            else:
                # Polygon / linestrip / etc.
                coords = []
                for p in points:
                    coords.append(f"{max(0.0, min(1.0, p.x() / w)):.6f}")
                    coords.append(f"{max(0.0, min(1.0, p.y() / h)):.6f}")
                lines.append(f"{class_id} {' '.join(coords)}")

        # Write label file
        os.makedirs(label_dir, exist_ok=True)
        with open(label_file, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        # Persist any newly added labels back to .labels.json
        if new_labels_added and os.path.exists(labels_json):
            labels_data["labels"] = class_list
            with open(labels_json, "w", encoding="utf-8") as f:
                _json.dump(labels_data, f, indent=2)

        self.setClean()
        self.statusBar().showMessage(f"✓ Saved annotations to {os.path.basename(label_file)}", 8000)

    def _saveFile(self, filename):
        """Internal save with error handling"""
        try:
            if not filename:
                return
            
            # Check disk space before saving
            try:
                import shutil
                stat = shutil.disk_usage(os.path.dirname(filename))
                free_mb = stat.free / (1024 * 1024)
                if free_mb < 10:  # Less than 10 MB free
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Low Disk Space",
                        f"Warning: Only {free_mb:.1f} MB of disk space remaining.\n\n"
                        f"Save operation will continue, but you should free up space soon."
                    )
            except Exception as e:
                print(f"Warning: Could not check disk space: {e}")
            
            if self.saveLabels(filename):
                self.addRecentFile(filename)
                self.setClean()
                self.statusBar().showMessage(f"✓ Saved to {os.path.basename(filename)}", 8000)
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Save Failed",
                    f"Failed to save labels to:\n{filename}\n\n"
                    f"Please check the console for error details."
                )
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Save Error",
                f"Error saving file:\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

    def closeFile(self, _value=False):
        self._explore_image_list = [] 
        if not self.mayContinue():
            return
        self.resetState()
        self.actions.autoLabel.setEnabled(False)
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)  # type: ignore[attr-defined]

    def getLabelFile(self):
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + ".json"

        return label_file

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this image and its label file. Proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        # If in explore mode, delete both image and label file
        if hasattr(self, "_explore_image_list") and self._explore_image_list:
            filename = self.filename
            # Detect split from path
            if "/train/" in filename or "\\train\\" in filename:
                split = "train"
            elif "/val/" in filename or "\\val\\" in filename:
                split = "val"
            else:
                split = None

            # Delete image file
            if os.path.exists(filename):
                os.remove(filename)
                logger.info("Image file is removed: {}".format(filename))

            # Delete label file
            if split:
                dataset_path = self.dataset_path
                img_name = os.path.splitext(os.path.basename(filename))[0]
                label_dir = os.path.join(dataset_path, "labels", split)
                label_file = os.path.join(label_dir, img_name + ".txt")
                if os.path.exists(label_file):
                    os.remove(label_file)
                    logger.info("Label file is removed: {}".format(label_file))

            # Remove from explore image list and refresh
            if filename in self._explore_image_list:
                self._explore_image_list.remove(filename)
            
            # Load next image if available, otherwise close explore mode
            if self._explore_image_list:
                # Load the next image (first in the updated list)
                next_file = self._explore_image_list[0]
                self.set_explore_file_list(self._explore_image_list, next_file)
                self.loadFile(next_file)
            else:
                # No more files, close explore mode
                self.statusBar().showMessage("No more images in explore mode", 8000)
                self.closeFile()
            return

        # Otherwise, delete only the label file (normal mode)
        label_file = self.getLabelFile()
        if os.path.exists(label_file):
            os.remove(label_file)
            logger.info("Label file is removed: {}".format(label_file))

            item = self.fileListWidget.currentItem()
            if item:
                item.setCheckState(Qt.Unchecked)
                self._set_file_item_color(item, False)

            self.resetState()

    def deleteImageFile(self):
        if not self.filename or not os.path.exists(self.filename):
            return
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this image file. Proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        filename = self.filename

        # Find index before removing from list
        image_list = self.imageList
        curr_index = image_list.index(filename) if filename in image_list else -1

        # Remove from file list widget
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            if item and item.text() == filename:
                self.fileListWidget.takeItem(i)
                break

        # Delete the image file from disk
        os.remove(filename)
        logger.info("Image file is removed: {}".format(filename))

        # Navigate to next image or close
        updated_list = self.imageList
        if updated_list:
            next_index = min(curr_index, len(updated_list) - 1)
            self.filename = None
            self.loadFile(updated_list[next_index])
        else:
            self.closeFile()

    # Message Dialogs. #
    def hasLabels(self):
        if self.noShapes():
            self.errorMessage(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(self.filename)
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def onVerifiedEmptyChanged(self, state):
        """Handle verified empty checkbox state changes."""
        self.verified_empty = state == Qt.Checked
        self.setDirty()
        
        # Auto-save when checked to create JSON file
        if self.verified_empty and self.filename:
            self.saveFile()
    
    def toggleKeepPrevMode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]

    def switchTaskType(self):
        """Switch task type using the full TaskTypeDialog (Segmentation / Bounding Box / OBB)."""
        dialog = TaskTypeDialog(self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted or not dialog.task_type:
            return
        new_type = dialog.task_type
        if new_type == self.task_type:
            return  # no change
        if not self.mayContinue():
            return

        self.task_type = new_type

        _mode_names = {"segmentation": "Segmentation", "bbox": "Bounding Box", "obb": "OBB"}
        mode = _mode_names.get(self.task_type, self.task_type)
        self.setWindowTitle(f"{__appname__} - No dataset - Mode: {mode}")

        # Enable/disable drawing modes for the new task type
        self.actions.createMode.setEnabled(self.task_type == "segmentation")
        self.actions.createRectangleMode.setEnabled(self.task_type in ("bbox", "obb"))

        # Clear current file and reset state
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)

        # Clear dataset and model — they were for the old task type
        self.yolo_model = None
        self.dataset_path = None
        self.updateAutoLabelAvailability()

        # Persist the new task type so the app reopens in the same mode
        self.settings.setValue("task_type", self.task_type)

        self.statusBar().showMessage(f"✓ Switched to {mode} mode", 5000)

    def removeSelectedPoint(self):
        self.canvas.removeSelectedPoint()
        self.canvas.update()
        if not self.canvas.hShape.points:  # type: ignore[union-attr]
            self.canvas.deleteShape(self.canvas.hShape)
            self.remLabels([self.canvas.hShape], strict=False)
            if self.noShapes():
                for action in self.actions.onShapesPresent:  # type: ignore[attr-defined]
                    action.setEnabled(False)
        self.setDirty()

    def deleteSelectedShape(self):
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = self.tr(
            "You are about to permanently delete {} polygons, " "proceed anyway?"
        ).format(len(self.canvas.selectedShapes))
        if yes == QtWidgets.QMessageBox.warning(
            self, self.tr("Attention"), msg, yes | no, yes
        ):
            self.remLabels(self.canvas.deleteSelected(), strict=False)
            self.setDirty()
            if self.noShapes():
                for action in self.actions.onShapesPresent:  # type: ignore[attr-defined]
                    action.setEnabled(False)

    def copyShape(self):
        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def openDirDialog(self, _value=False, dirpath=None):
        """Open directory with error handling"""
        try:
            self._explore_image_list = []
            if not self.mayContinue():
                return

            defaultOpenDirPath = dirpath if dirpath else "."
            if self.lastOpenDir and osp.exists(self.lastOpenDir):
                defaultOpenDirPath = self.lastOpenDir
            else:
                defaultOpenDirPath = osp.dirname(self.filename) if self.filename else "."

            targetDirPath = str(
                QtWidgets.QFileDialog.getExistingDirectory(
                    self,
                    self.tr("%s - Open Directory") % __appname__,
                    defaultOpenDirPath,
                    QtWidgets.QFileDialog.ShowDirsOnly
                    | QtWidgets.QFileDialog.DontResolveSymlinks,
                )
            )
            
            if not targetDirPath:
                return
            
            # Validate directory
            if not os.path.exists(targetDirPath):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Directory Not Found",
                    f"The selected directory does not exist:\n{targetDirPath}"
                )
                return
            
            # Check read permissions
            if not os.access(targetDirPath, os.R_OK):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Permission Denied",
                    f"No read permission for directory:\n{targetDirPath}\n\n"
                    f"Please check folder permissions."
                )
                return
            
            self.importDirImages(targetDirPath)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error Opening Directory",
                f"An error occurred while opening the directory:\n\n"
                f"{type(e).__name__}: {str(e)}"
            )
            import traceback
            traceback.print_exc()

    @property
    def imageList(self):
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            lst.append(item.text())  # type: ignore[union-attr]
        return lst

    @property
    def imageList2(self):
        # This will hold the explored images
        return getattr(self, "_explore_image_list", [])

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(tuple(extensions)):
                continue
            label_file = osp.splitext(file)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # type: ignore[attr-defined]
            has_label = QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file)
            item.setCheckState(Qt.Checked if has_label else Qt.Unchecked)  # type: ignore[attr-defined]
            self._set_file_item_color(item, has_label)
            self.fileListWidget.addItem(item)

        if len(self.imageList) > 1:
            self.actions.openNextImg.setEnabled(True)  # type: ignore[attr-defined]
            self.actions.openPrevImg.setEnabled(True)  # type: ignore[attr-defined]

        self.openNextImg()

    def _set_file_item_color(self, item, annotated):
        """Color a file-list item green when annotated, grey when not."""
        if annotated:
            item.setForeground(QtGui.QBrush(QtGui.QColor("#4CAF50")))
        else:
            item.setForeground(QtGui.QBrush(QtGui.QColor("#888888")))

    def importDirImages(self, dirpath, pattern=None, load=True):
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)
        self._explore_image_list = []

        if not self.mayContinue() or not dirpath:
            return

        if self.dataset_path:
            # Use labels from dataset; prompt to create if none exist yet
            label_names = self.load_label_list_json(self.dataset_path)
            if label_names is None:
                # Dataset folder has no .labels.json — prompt to create classes
                label_dialog = LabelListDialog(self)
                label_dialog.setWindowTitle("Create Class Labels for Dataset")
                if label_dialog.exec_() == QtWidgets.QDialog.Accepted:
                    label_names = label_dialog.get_labels()
                else:
                    label_names = []
                # Persist the newly created labels
                if label_names:
                    _lj_path = os.path.join(self.dataset_path, ".labels.json")
                    try:
                        with open(_lj_path, "w", encoding="utf-8") as _f:
                            json.dump({"labels": label_names, "type": self.task_type}, _f, ensure_ascii=False, indent=2)
                    except Exception as _e:
                        logger.error(f"Failed to write .labels.json: {_e}")
            else:
                logger.info(f"Loaded label list from {os.path.join(self.dataset_path, '.labels.json')}")
        else:
            # No dataset loaded, prompt for labels
            label_names = self.load_label_list_json(dirpath)
            if label_names is None:
                label_dialog = LabelListDialog(self)
                if label_dialog.exec_() == QtWidgets.QDialog.Accepted:
                    label_names = label_dialog.get_labels()
                else:
                    label_names = []
            else:
                logger.info(f"Loaded label list from {os.path.join(dirpath, '.labels.json')}")

        self.uniqLabelList.clear()
        for label in label_names:
            item = self.uniqLabelList.createItemFromLabel(label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(label)
            self.uniqLabelList.setItemLabel(item, label, rgb)
        self.labelDialog.setLabels(label_names)
        self.updateMoveButtons()
        self.lastOpenDir = dirpath

        # --- Save .labels.json after uniqLabelList is populated ---
        if label_names:
            self.save_label_list_json(dirpath)
        # ---------------------------------------------------------

        self.filename = None
        self.fileListWidget.clear()

        filenames = self.scanAllImages(dirpath)
        if pattern:
            try:
                filenames = [f for f in filenames if re.search(pattern, f)]
            except re.error:
                pass
        for filename in filenames:
            label_file = osp.splitext(filename)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # type: ignore[attr-defined]
            has_label = QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file)
            item.setCheckState(Qt.Checked if has_label else Qt.Unchecked)  # type: ignore[attr-defined]
            self._set_file_item_color(item, has_label)
            self.fileListWidget.addItem(item)
        self.openNextImg(load=load)

    def scanAllImages(self, folderPath):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        images = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.normpath(osp.join(root, file))
                    images.append(relativePath)
        images = natsort.os_sorted(images)
        return images
    
    def start_yolo_training(self):
        """Start YOLO training with comprehensive error handling"""
        try:
            # Safe check for running thread - ignore if it's marked as cancelled
            try:
                if hasattr(self, "training_thread") and self.training_thread:
                    is_cancelling = getattr(self, '_cancelling_training', False)
                    if self.training_thread.isRunning():
                        if is_cancelling:
                            QtWidgets.QMessageBox.information(
                                self,
                                "Please Wait",
                                "Training is still stopping. Please try again in a moment."
                            )
                        else:
                            QtWidgets.QMessageBox.warning(
                                self,
                                "Training Already Running",
                                "A training session is already in progress.\n\n"
                                "Please wait for it to complete or cancel it first."
                            )
                        return
                    else:
                        # Thread finished — safe to clear
                        self.training_thread = None
            except RuntimeError:
                self.training_thread = None

            # Check if dataset path is set
            if not self.dataset_path:
                QtWidgets.QMessageBox.warning(
                    self, 
                    "No Dataset Selected",
                    "Please select a dataset folder before training.\n\n"
                    "Steps:\n"
                    "1. Click 'Select Project' button\n"
                    "2. Choose or create a properly formatted YOLO dataset\n"
                    "3. Export your annotations (Project → Export YOLO)\n"
                    "4. Then start training"
                )
                return
            
            # Validate dataset exists
            if not os.path.exists(self.dataset_path):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Dataset Not Found",
                    f"The selected dataset path does not exist:\n{self.dataset_path}\n\n"
                    f"Please select a valid dataset."
                )
                self.dataset_path = None
                return
            
            # Check for data.yaml
            data_yaml = os.path.join(self.dataset_path, "data.yaml")
            if not os.path.exists(data_yaml):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid Dataset",
                    f"data.yaml not found in:\n{self.dataset_path}\n\n"
                    f"Please export your dataset first (Project → Export YOLO)"
                )
                return
            
            # Verify data.yaml is valid
            try:
                import yaml
                with open(data_yaml, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if not yaml_data:
                        raise ValueError("data.yaml is empty")
                    required_keys = ['train', 'val', 'nc', 'names']
                    missing = [k for k in required_keys if k not in yaml_data]
                    if missing:
                        raise ValueError(f"Missing required keys: {', '.join(missing)}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid Dataset Configuration",
                    f"Error reading data.yaml:\n{str(e)}\n\n"
                    f"Please re-export your dataset."
                )
                return
            
            # Check disk space
            try:
                import shutil
                stat = shutil.disk_usage(self.dataset_path)
                free_gb = stat.free / (1024**3)
                if free_gb < 5:
                    response = QtWidgets.QMessageBox.question(
                        self,
                        "Low Disk Space",
                        f"Warning: Only {free_gb:.1f} GB of disk space remaining.\n\n"
                        f"Training requires significant disk space for:\n"
                        f"• Model checkpoints\n"
                        f"• Training logs\n"
                        f"• Cached data\n\n"
                        f"Continue anyway?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    )
                    if response == QtWidgets.QMessageBox.No:
                        return
            except Exception as e:
                print(f"Warning: Could not check disk space: {e}")

            # Full dataset validation before opening parameters dialog
            is_valid, val_error = validate_yolo_dataset(self.dataset_path)
            if not is_valid:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Dataset Validation Failed",
                    f"{val_error}\n\nPlease fix the dataset issues before training."
                )
                return

            # Warn if there are very few labeled training images
            train_labels_dir = os.path.join(self.dataset_path, "labels", "train")
            if os.path.isdir(train_labels_dir):
                n_train = sum(
                    1 for f in os.listdir(train_labels_dir)
                    if f.endswith(".txt") and os.path.getsize(
                        os.path.join(train_labels_dir, f)) > 0
                )
                if n_train < 8:
                    reply = QtWidgets.QMessageBox.warning(
                        self,
                        "Very Few Training Images",
                        f"Only {n_train} labeled training image(s) found.\n\n"
                        "YOLO typically needs at least 8–10 labeled images per class "
                        "to train usefully. Training with fewer images may produce a "
                        "model that doesn't generalise.\n\n"
                        "Continue anyway?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.No,
                    )
                    if reply != QtWidgets.QMessageBox.Yes:
                        return

            # Get training parameters
            if self.task_type == "bbox":
                dialog = YoloTrainDialog_bbx(parent=self)
                ResultsDialogClass = TrainingResultsDialog_bbx
                WorkerClass = TrainingWorker_bbx
                ProgressDialogClass = TrainingDialog_bbx
            elif self.task_type == "obb":
                dialog = YoloTrainDialog_obb(parent=self)
                ResultsDialogClass = TrainingResultsDialog_obb
                WorkerClass = TrainingWorker_obb
                ProgressDialogClass = TrainingDialog_obb
            else:
                dialog = YoloTrainDialog(parent=self)
                ResultsDialogClass = TrainingResultsDialog
                WorkerClass = TrainingWorker
                ProgressDialogClass = TrainingDialog

            if dialog.exec_() != QtWidgets.QDialog.Accepted:
                return
            
            params = dialog.get_params()
            self.status(f"Training parameters: {params}")
            
            # Validate parameters
            if params.get('epochs', 0) <= 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Parameters",
                    "Epochs must be greater than 0."
                )
                return
            
            if params.get('batch_size', 0) <= 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Parameters",
                    "Batch size must be greater than 0."
                )
                return
            
            # Create progress dialog
            try:
                self.progress_dialog = ProgressDialogClass(params["epochs"], parent=self)
                self.progress_dialog.cancelled_signal.connect(self.cancel_training)
                self.progress_dialog.show()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Dialog Error",
                    f"Failed to create progress dialog:\n{str(e)}"
                )
                return

            # Create worker thread
            try:
                self.training_thread = QThread()
                self.worker = WorkerClass(data_yaml, params)
                self.worker.moveToThread(self.training_thread)
                
                # Connect signals
                self.training_thread.started.connect(self.worker.run)
                self.worker.progress_signal.connect(self.progress_dialog.update_metrics)
                self.worker.status_signal.connect(self.status)
                self.worker.status_signal.connect(self.progress_dialog.set_status)
                self.worker.finished_signal.connect(self.on_training_finished)
                self.worker.finished_signal.connect(self.training_thread.quit)
                self.worker.finished_signal.connect(self.worker.deleteLater)
                self.worker.error_signal.connect(self.show_training_error)
                self.training_thread.finished.connect(self.training_thread.deleteLater)
                
                # Start training
                self.training_thread.start()
                
                # Save ResultsDialogClass for later
                self._training_results_dialog_class = ResultsDialogClass
                
                print(f"[TRAINING] Started with parameters: {params}")
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Training Start Error",
                    f"Failed to start training:\n\n{str(e)}"
                )
                if hasattr(self, 'progress_dialog') and self.progress_dialog:
                    self.progress_dialog.close()
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Unexpected Error",
                f"An unexpected error occurred:\n\n"
                f"{type(e).__name__}: {str(e)}\n\n"
                f"Please check the console for details."
            )
            import traceback
            traceback.print_exc()

    def on_training_finished(self, results_text):
        # Close progress dialog
        if hasattr(self, "progress_dialog") and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        # Ensure thread is stopped before clearing reference
        if hasattr(self, "training_thread") and self.training_thread:
            finished = self.training_thread.wait(3000)
            if finished:
                self.training_thread = None
            else:
                print("[DEBUG] Training thread did not finish in time, not clearing reference.")
        self.worker = None

        # Always reset cancellation flag so future cancels work
        self._cancelling_training = False

        # Only show results dialog for normal completion
        if results_text == "__cancelled__":
            self.status("Training cancelled by user.")
        else:
            dlg_class = getattr(self, "_training_results_dialog_class", TrainingResultsDialog)
            dlg = dlg_class(results_text, self)
            dlg.exec_()
            # Auto-load the newly trained weights so the user can start
            # auto-labeling immediately without manually reloading the model.
            if self.dataset_path:
                self.load_yolo_model(show_message=False)

    def cancel_training(self):
        # Prevent multiple calls
        if getattr(self, '_cancelling_training', False):
            return
        self._cancelling_training = True

        # Update the progress dialog in place — no second popup
        if hasattr(self, "progress_dialog") and self.progress_dialog:
            try:
                self.progress_dialog.setWindowTitle("Cancelling Training…")
                self.progress_dialog.cancel_btn.setText("Cancelling, please wait…")
                self.progress_dialog.cancel_btn.setEnabled(False)
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass

        # Signal the worker to stop
        if hasattr(self, "worker") and self.worker:
            try:
                self.worker.stop()
            except (RuntimeError, AttributeError):
                pass

        # Ask the thread to quit; poll without blocking the event loop
        if hasattr(self, "training_thread") and self.training_thread:
            try:
                self.training_thread.quit()
            except Exception:
                pass

        # Use a poll timer so the UI stays responsive while we wait
        self._cancel_poll_count = 0
        self._cancel_poll_timer = QtCore.QTimer(self)
        self._cancel_poll_timer.setInterval(200)
        self._cancel_poll_timer.timeout.connect(self._poll_cancel_thread)
        self._cancel_poll_timer.start()

    def _poll_cancel_thread(self):
        """Called every 200 ms while waiting for the training thread to exit."""
        MAX_POLLS = 25  # 5 seconds — subprocess kill is instant
        self._cancel_poll_count += 1
        thread_done = True
        if hasattr(self, "training_thread") and self.training_thread:
            try:
                thread_done = self.training_thread.wait(0)
                if thread_done:
                    self.training_thread = None
            except Exception:
                self.training_thread = None

        if thread_done or self._cancel_poll_count >= MAX_POLLS:
            self._cancel_poll_timer.stop()
            # Close the progress dialog now that the thread is done
            if hasattr(self, "progress_dialog") and self.progress_dialog:
                try:
                    self.progress_dialog.close()
                    self.progress_dialog = None
                except Exception:
                    pass
            self.worker = None
            self.status("Training cancelled.")
            self._cancelling_training = False

    def start_yolo_retraining(self):
        # Safe check for running thread (handle case where thread object was deleted)
        try:
            if hasattr(self, "training_thread") and self.training_thread:
                is_cancelling = getattr(self, '_cancelling_training', False)
                if self.training_thread.isRunning():
                    if is_cancelling:
                        QtWidgets.QMessageBox.information(
                            self, "Please Wait",
                            "Training is still stopping. Please try again in a moment."
                        )
                    else:
                        QtWidgets.QMessageBox.warning(self, "Training", "Training is already running!")
                    return
                else:
                    # Thread finished — safe to clear
                    self.training_thread = None
        except RuntimeError:
            # Thread object was deleted, clear the reference
            self.training_thread = None

        # Check if dataset path is set
        if not self.dataset_path:
            QtWidgets.QMessageBox.warning(
                self, 
                "No Dataset Selected",
                "Please select a dataset folder before retraining.\n\n"
                "Use the 'Select Dataset Folder' button to choose a properly formatted YOLO dataset."
            )
            return

        if self.task_type == "bbox":
            dialog = YoloRetrainDialog_bbx(dataset_path=self.dataset_path, parent=self)
            ResultsDialogClass = RetrainResultsDialog_bbx
        elif self.task_type == "obb":
            dialog = YoloRetrainDialog_obb(dataset_path=self.dataset_path, parent=self)
            ResultsDialogClass = RetrainResultsDialog_obb
        else:
            dialog = YoloRetrainDialog(dataset_path=self.dataset_path, parent=self)
            ResultsDialogClass = RetrainResultsDialog

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            params = dialog.get_params()
            self.status(f"Retraining parameters selected: {params}")
            data_yaml = os.path.join(self.dataset_path, "data.yaml")

            if self.task_type == "bbox":
                self.progress_dialog = RetrainTrainingDialog_bbx(params["epochs"], parent=self)
                WorkerClass = RetrainWorker_bbx
            elif self.task_type == "obb":
                self.progress_dialog = RetrainTrainingDialog_obb(params["epochs"], parent=self)
                WorkerClass = RetrainWorker_obb
            else:
                self.progress_dialog = RetrainTrainingDialog(params["epochs"], parent=self)
                WorkerClass = RetrainWorker

            self.progress_dialog.cancelled_signal.connect(self.cancel_training)
            self.progress_dialog.show()

            self.training_thread = QThread()
            self.worker = WorkerClass(data_yaml, params)
            self.worker.moveToThread(self.training_thread)
            self.worker.error_signal.connect(self.show_training_error)

            self.training_thread.started.connect(self.worker.run)
            self.worker.progress_signal.connect(self.progress_dialog.update_metrics)
            self.worker.status_signal.connect(self.status)
            self.worker.status_signal.connect(self.progress_dialog.set_status)
            self.worker.finished_signal.connect(self.on_retraining_finished)
            self.worker.finished_signal.connect(self.training_thread.quit)
            self.worker.finished_signal.connect(self.worker.deleteLater)
            
            self.training_thread.finished.connect(self.training_thread.deleteLater)

            self.training_thread.start()
            
            self._retraining_results_dialog_class = ResultsDialogClass

    def on_retraining_finished(self, results_text):
        # Close progress dialog
        if hasattr(self, "progress_dialog") and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        # Ensure thread is stopped before clearing reference
        if hasattr(self, "training_thread") and self.training_thread:
            finished = self.training_thread.wait(3000)
            if finished:
                self.training_thread = None
            else:
                print("[DEBUG] Training thread did not finish in time, not clearing reference.")

        # Reset cancellation flag if it was set
        if hasattr(self, '_cancelling_training'):
            self._cancelling_training = False

        # Only show results dialog if not cancelled
        if results_text == "__cancelled__":
            self.status("Training cancelled by user.")
        else:
            self.status("Training finished.")
            dlg_class = getattr(self, "_retraining_results_dialog_class", RetrainResultsDialog)
            dlg = dlg_class(results_text, self)
            dlg.setWindowModality(QtCore.Qt.ApplicationModal)
            dlg.exec_()
            # Auto-load the newly retrained weights
            if self.dataset_path:
                self.load_yolo_model(show_message=False)

    def explore_yolo_dialog(self):
        dataset_root = self.dataset_path
        if not dataset_root:
            QtWidgets.QMessageBox.warning(self, "Explore Dataset", "Please set the dataset path first.")
            return
        
        # Show loading dialog
        progress = QtWidgets.QProgressDialog("Loading dataset...", None, 0, 0, self)
        progress.setWindowTitle("Loading Dataset")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        utils.fit_dialog(progress, 300, 100)
        progress.setValue(0)
        progress.show()
        QtWidgets.QApplication.processEvents()
        
        try:
            # Create dialog but don't show it yet
            dlg = ExploreYoloDialog(self, dataset_root, self.loadFile, progress)
            # Dialog will show itself when loading completes
            dlg.exec_()
        except Exception as e:
            progress.close()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
            raise

    def updateProjectLabelList(self):
        self.projectLabelList.clear()
        if self.dataset_path:
            label_file = os.path.join(self.dataset_path, ".labels.json")
            if os.path.exists(label_file):
                try:
                    with open(label_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    labels = data.get("labels", [])
                    for label in labels:
                        item = QtWidgets.QListWidgetItem(label)
                        r, g, b = self._get_rgb_by_label(label)
                        item.setForeground(QtGui.QColor(r, g, b))
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
                        item.setToolTip(label)
                        self.projectLabelList.addItem(item)
                except Exception as e:
                    self.projectLabelList.addItem(f"Error loading labels: {e}")
            else:
                self.projectLabelList.addItem("No .labels.json found in dataset path")
        else:
            self.projectLabelList.addItem("No dataset selected")

    def _load_explore_yolo_labels(self, img_path):
        """Load YOLO .txt labels for *img_path* when navigating an exported dataset.

        The dataset root is inferred from the image path itself:
        …/images/train/img.jpg  →  dataset root is the parent of `images/`.
        Works for both train and val splits.
        """
        norm = img_path.replace("\\", "/")
        dataset_path = None
        split = None
        for sp in ("train", "val"):
            marker = f"/images/{sp}/"
            idx = norm.find(marker)
            if idx >= 0:
                dataset_path = norm[:idx].replace("/", os.sep)
                split = sp
                break

        if not dataset_path:
            return

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_file = os.path.join(dataset_path, "labels", split, img_name + ".txt")
        if not os.path.exists(label_file):
            return

        labels_json = os.path.join(dataset_path, ".labels.json")
        class_list = []
        is_obb = (self.task_type == "obb")
        if os.path.exists(labels_json):
            try:
                with open(labels_json, "r", encoding="utf-8") as f:
                    ldata = json.load(f)
                    class_list = ldata.get("labels", [])
                    if ldata.get("type"):
                        is_obb = ldata.get("type") == "obb"
            except Exception:
                pass

        image = QtGui.QImage(img_path)
        w, h = image.width(), image.height()
        if w == 0 or h == 0:
            return

        shapes = []
        try:
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    label = class_list[class_id] if class_id < len(class_list) else str(class_id)
                    coords = [float(x) for x in parts[1:]]

                    other_data = {}
                    if len(coords) == 4:
                        cx, cy, bw, bh = coords
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        points = [(x1, y1), (x2, y2)]
                        shape_type = "rectangle"
                    elif len(coords) == 8 and is_obb:
                        corners = [(coords[i] * w, coords[i + 1] * h) for i in range(0, 8, 2)]
                        # Estimate angle from first edge direction
                        dx = corners[1][0] - corners[0][0]
                        dy = corners[1][1] - corners[0][1]
                        angle_deg = math.degrees(math.atan2(dy, dx))
                        other_data = {"rotation": angle_deg}
                        # Use actual 4 corners directly (loadLabels will add handle as 5th point)
                        points = corners
                        shape_type = "rotation"
                    else:
                        points = [
                            (coords[i] * w, coords[i + 1] * h)
                            for i in range(0, len(coords), 2)
                        ]
                        shape_type = "polygon"

                    shapes.append({
                        "label":      label,
                        "points":     points,
                        "shape_type": shape_type,
                        "flags":      {},
                        "group_id":   None,
                        "description": "",
                        "other_data": other_data,
                        "mask":       None,
                    })
        except Exception:
            return

        if shapes:
            self.loadLabels(shapes)

    def set_explore_file_list(self, file_list, current_file):

        self._explore_image_list = file_list[:]  # Store explored images
        self.fileListWidget.clear()
        for filename in file_list:
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            # Check for corresponding YOLO .txt label file
            txt_label = osp.splitext(filename)[0].replace(
                osp.sep + "images" + osp.sep,
                osp.sep + "labels" + osp.sep,
            ) + ".txt"
            self._set_file_item_color(item, osp.exists(txt_label))
            self.fileListWidget.addItem(item)
        if current_file in file_list:
            self.fileListWidget.setCurrentRow(file_list.index(current_file))
            self.filename = current_file
            self.loadFile(current_file)
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)
        self.actions.deleteFile.setEnabled(True)

    def _load_pretrained_yolo26(self):
        """Load a pretrained YOLO26 nano model for auto-labeling.

        Used as fallback when no custom-trained weights exist.
        Ultralytics auto-downloads the weights (~6 MB) on first use.
        """
        if self.task_type == "segmentation":
            model_name = "yolo26n-seg.pt"
        elif self.task_type == "obb":
            model_name = "yolo26n-obb.pt"
        else:
            model_name = "yolo26n.pt"
        print(f"[INFO] No custom weights found — loading pretrained {model_name}")
        loading_dialog = ModelLoadingDialog(self)
        loading_dialog.set_model_info(
            model_name,
            self.task_type,
            ["pretrained on COCO — train your own model for custom classes"],
        )
        loading_dialog.load_model(model_name)
        loading_dialog.exec_()
        if loading_dialog.error_message:
            if "internet" in loading_dialog.error_message.lower() or "connect" in loading_dialog.error_message.lower():
                msg = (
                    f"Could not download pretrained {model_name}.\n\n"
                    "An internet connection is required the first time.\n"
                    "After download the model is cached locally."
                )
            else:
                msg = f"Could not load pretrained {model_name}:\n{loading_dialog.error_message}"
            QtWidgets.QMessageBox.warning(self, "Pretrained YOLO26 Unavailable", msg)
            self.yolo_model = None
        else:
            self.yolo_model = loading_dialog.model
            self.statusBar().showMessage(
                f"✓ Pretrained {model_name} ready for auto-labeling", 5000
            )
            print(f"[SUCCESS] Loaded pretrained {model_name}")
            self.updateAutoLabelAvailability()

    def load_yolo_model(self, show_message=True):
        """Load YOLO model with comprehensive error handling"""
        import json
        
        try:
            # Clean up previous model to prevent memory leak
            if hasattr(self, 'yolo_model') and self.yolo_model is not None:
                print("[DEBUG] Cleaning up previous model to free memory...")
                del self.yolo_model
                self.yolo_model = None
                
                # Clear GPU cache if using CUDA
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("[DEBUG] GPU cache cleared")
                except Exception as e:
                    print(f"[DEBUG] Could not clear GPU cache: {e}")
            
            # Validate dataset path
            if not self.dataset_path:
                if show_message:
                    QtWidgets.QMessageBox.warning(
                        self, 
                        "No Dataset Selected",
                        "Please select a dataset first before loading a model."
                    )
                self.yolo_model = None
                return
            
            if not os.path.exists(self.dataset_path):
                if show_message:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Dataset Not Found",
                        f"The selected dataset path does not exist:\n{self.dataset_path}\n\n"
                        f"Please select a valid dataset."
                    )
                self.yolo_model = None
                self.dataset_path = None
                return
            
            dataset_name = os.path.basename(self.dataset_path)
            weights_root = os.path.join(self.dataset_path, "weight")
            trained_status = "Not trained"
            weights_path = None

            # Check if weights folder exists
            if not os.path.exists(weights_root):
                mode = {"segmentation": "Segmentation", "bbox": "Bounding Box", "obb": "OBB"}.get(self.task_type, self.task_type)
                self.setWindowTitle(f"{__appname__} - Dataset: {dataset_name} [{trained_status}] - Mode: {mode}")
                self._load_pretrained_yolo26()
                return

            # Find latest trained model
            try:
                exp_folders = [d for d in os.listdir(weights_root) 
                             if d.startswith("exp") and os.path.isdir(os.path.join(weights_root, d))]
                
                if exp_folders:
                    exp_folders_sorted = sorted(exp_folders, key=lambda x: int(x[3:]) if x[3:].isdigit() else 0, reverse=True)
                    for exp_folder in exp_folders_sorted:
                        candidate = os.path.join(weights_root, exp_folder, "best.pt")
                        if os.path.exists(candidate):
                            # Verify file is not empty and is readable
                            if os.path.getsize(candidate) == 0:
                                print(f"Warning: {candidate} is empty, skipping...")
                                continue
                            weights_path = candidate
                            trained_status = "Trained"
                            break
            except (OSError, PermissionError) as e:
                if show_message:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "File System Error",
                        f"Error accessing weights folder:\n{str(e)}\n\n"
                        f"Please check folder permissions."
                    )
                self.yolo_model = None
                return

            # Update window title
            _mn = {"segmentation": "Segmentation", "bbox": "Bounding Box", "obb": "OBB"}
            mode = _mn.get(self.task_type, self.task_type)
            self.setWindowTitle(f"{__appname__} - Dataset: {dataset_name} [{trained_status}] - Mode: {mode}")

            if not weights_path:
                self._load_pretrained_yolo26()
                return

            # Verify weights file
            try:
                file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
                if file_size_mb < 1:  # Suspiciously small model
                    if show_message:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Invalid Model File",
                            f"The model file appears to be corrupted or incomplete:\n{weights_path}\n\n"
                            f"File size: {file_size_mb:.2f} MB (expected > 1 MB)\n\n"
                            f"Please retrain the model."
                        )
                    self.yolo_model = None
                    return
            except (OSError, PermissionError) as e:
                if show_message:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "File Access Error",
                        f"Cannot access model file:\n{str(e)}\n\n"
                        f"Please check file permissions."
                    )
                self.yolo_model = None
                return

            # Get task type and classes from .labels.json
            task_type = "Unknown"
            classes = []
            labels_json_path = os.path.join(self.dataset_path, ".labels.json")

            if os.path.exists(labels_json_path):
                try:
                    with open(labels_json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        task_type = data.get("type", "Unknown")
                        classes = data.get("labels", [])

                        if not classes:
                            print("Warning: No classes found in .labels.json")

                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON in .labels.json: {e}")
                    if show_message:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Invalid Configuration",
                            f"The .labels.json file is corrupted or invalid.\n\n"
                            f"Error: {str(e)}\n\n"
                            f"Model will still load but class names may not be available."
                        )
                except Exception as e:
                    print(f"Error reading .labels.json: {e}")

            # Reject dataset/model if it doesn't match the current task type
            if task_type != "Unknown" and task_type != self.task_type:
                _mn = {"segmentation": "Segmentation", "bbox": "Bounding Box", "obb": "OBB"}
                current_mode = _mn.get(self.task_type, self.task_type)
                dataset_mode = _mn.get(task_type, task_type)
                print(f"[WARNING] Dataset task type '{task_type}' does not match current mode '{self.task_type}' — skipping load")
                self.dataset_path = None
                self.yolo_model = None
                if show_message:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Wrong Task Type",
                        f"The selected dataset was trained for '{dataset_mode}' mode,\n"
                        f"but the app is running in '{current_mode}' mode.\n\n"
                        f"Please select a dataset that matches the current mode."
                    )
                return

            # Create and show loading dialog
            loading_dialog = ModelLoadingDialog(self)
            loading_dialog.set_model_info(weights_path, task_type, classes)
            
            # Start loading model in background
            loading_dialog.load_model(weights_path)
            
            # Show dialog (blocks until loading is complete)
            loading_dialog.exec_()
            
            # Get the loaded model
            if loading_dialog.error_message:
                error_msg = loading_dialog.error_message

                # Special case: old YOLOv5 weights detected — warn then fall back to pretrained
                if error_msg.startswith("Old YOLOv5 Model Detected"):
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Old YOLOv5 Model — Loading Pretrained YOLO26",
                        error_msg + "\n\nLoading pretrained YOLO26 model instead.",
                    )
                    self._load_pretrained_yolo26()
                    return

                # Provide specific guidance based on error type
                if "out of memory" in error_msg.lower():
                    guidance = (
                        "\n\nSuggestions:\n"
                        "• Close other applications to free up memory\n"
                        "• Restart the application\n"
                        "• Try using a smaller model (e.g., yolo26n instead of yolo26x)"
                    )
                elif "cuda" in error_msg.lower() or "gpu" in error_msg.lower():
                    guidance = (
                        "\n\nSuggestions:\n"
                        "• Update GPU drivers\n"
                        "• Check CUDA installation\n"
                        "• Model will automatically use CPU if GPU is unavailable"
                    )
                elif "file" in error_msg.lower() or "cannot load" in error_msg.lower():
                    guidance = (
                        "\n\nSuggestions:\n"
                        "• The model file may be corrupted\n"
                        "• Try retraining the model\n"
                        "• Check if the model was trained with a compatible PyTorch version"
                    )
                else:
                    guidance = (
                        "\n\nSuggestions:\n"
                        "• Retrain the model\n"
                        "• Check the console for detailed error messages\n"
                        "• Ensure PyTorch and dependencies are correctly installed"
                    )
                
                if show_message:
                    QtWidgets.QMessageBox.critical(
                        self, 
                        "Model Loading Failed", 
                        f"Failed to load YOLO model:\n\n{error_msg}{guidance}"
                    )
                self.yolo_model = None
            else:
                self.yolo_model = loading_dialog.model
                self.statusBar().showMessage(f"✓ Model loaded successfully from {os.path.basename(weights_path)}", 5000)
                print(f"[SUCCESS] Loaded YOLO model from {weights_path}")
                
                # Update auto-label availability (requires both model and image)
                self.updateAutoLabelAvailability()
                    
        except Exception as e:
            # Catch-all for unexpected errors
            if show_message:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Unexpected Error",
                    f"An unexpected error occurred while loading the model:\n\n"
                    f"{type(e).__name__}: {str(e)}\n\n"
                    f"Please check the console for detailed error information."
                )
            self.yolo_model = None
            import traceback
            traceback.print_exc()

    def auto_label_image_slot(self):
        """Open the floating AI panel in Auto Label mode."""
        if not hasattr(self, 'yolo_model') or self.yolo_model is None:
            self._load_pretrained_yolo26()
        if not hasattr(self, 'yolo_model') or self.yolo_model is None:
            return  # loading failed — error already shown by _load_pretrained_yolo26
        conf = getattr(self, "autoLabelThreshold", 0.35)
        self._ai_panel.sync_conf(conf)
        self._ai_panel.show_mode("auto")

    def _run_auto_label_from_panel(self, conf_threshold: float, append: bool = False):
        """Actually run auto-label — called by the panel's Run button."""
        import json

        try:
            # Check if model is loaded — auto-load pretrained YOLO26 if not
            if not hasattr(self, 'yolo_model') or self.yolo_model is None:
                self._load_pretrained_yolo26()
            if not hasattr(self, 'yolo_model') or self.yolo_model is None:
                return  # loading failed — error already shown

            # Check if image is loaded
            if not self.imagePath:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Image Loaded",
                    "Please open an image first before auto-labeling.\n\n"
                    "Use File → Open to load an image."
                )
                return
            
            if not os.path.exists(self.imagePath):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Image Not Found",
                    f"The image file no longer exists:\n{self.imagePath}\n\n"
                    f"It may have been moved or deleted."
                )
                return
            
            # Verify image is readable
            try:
                from PIL import Image
                test_img = Image.open(self.imagePath)
                test_img.verify()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid Image",
                    f"Cannot read the image file:\n{self.imagePath}\n\n"
                    f"Error: {str(e)}\n\n"
                    f"The file may be corrupted or in an unsupported format."
                )
                return
            
            self.autoLabelThreshold = conf_threshold  # keep in sync with dock slider

            # Load class list
            class_list = None
            if self.dataset_path:
                labels_json = os.path.join(self.dataset_path, ".labels.json")
                if os.path.exists(labels_json):
                    try:
                        with open(labels_json, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            class_list = data.get("labels", [])
                            
                            if not class_list:
                                QtWidgets.QMessageBox.warning(
                                    self,
                                    "No Classes Defined",
                                    "No classes are defined in the dataset.\n\n"
                                    "Auto-labeling may not work correctly."
                                )
                    except json.JSONDecodeError as e:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Configuration Error",
                            f"Error reading .labels.json:\n{str(e)}\n\n"
                            f"Auto-labeling will continue with default classes."
                        )
                    except Exception as e:
                        print(f"Warning: Error loading class list: {e}")
            
            # Run auto-labeling
            try:
                shape_dicts = auto_label_image(
                    self.yolo_model, 
                    self.imagePath, 
                    class_list=class_list, 
                    conf_threshold=conf_threshold
                )
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "cuda" in error_msg:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "GPU Out of Memory",
                        "GPU ran out of memory during inference.\n\n"
                        "Suggestions:\n"
                        "• Close other applications to free GPU memory\n"
                        "• Try with a smaller image\n"
                        "• Restart the application\n"
                        "• Use CPU mode (if available)"
                    )
                else:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Inference Error",
                        f"Error during model inference:\n\n{str(e)}\n\n"
                        f"The model may be incompatible with the image."
                    )
                return
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Auto-Label Error",
                    f"An error occurred during auto-labeling:\n\n"
                    f"{type(e).__name__}: {str(e)}\n\n"
                    f"Please check:\n"
                    f"• Image is valid and not corrupted\n"
                    f"• Model is compatible with the image\n"
                    f"• Sufficient system resources available"
                )
                import traceback
                traceback.print_exc()
                return
            
            if not shape_dicts:
                QtWidgets.QMessageBox.information(
                    self, 
                    "Auto Label", 
                    f"No objects detected with confidence ≥ {conf_threshold:.0%}\n\n"
                    f"Try:\n"
                    f"• Lowering the confidence threshold\n"
                    f"• Checking if the model is trained for these objects\n"
                    f"• Ensuring the image contains objects from your trained classes"
                )
                return

            # Convert detections to shapes
            shapes = []
            try:
                for shape_dict in shape_dicts:
                    shape = Shape(
                        label=shape_dict.get('label', ''),
                        shape_type=shape_dict.get('shape_type', 'polygon'),
                        group_id=shape_dict.get('group_id', None),
                        description=shape_dict.get('description', ''),
                        mask=shape_dict.get('mask', None)
                    )
                    for point in shape_dict.get('points', []):
                        shape.addPoint(QtCore.QPointF(point[0], point[1]))
                    shape.flags = shape_dict.get('flags', {})
                    shape.other_data = shape_dict.get('other_data', {})
                    if shape_dict.get('confidence') is not None:
                        shape.confidence = float(shape_dict['confidence'])
                    shape.close()
                    # Rotation shapes: ensure 5-point format (4 corners + handle)
                    if shape.shape_type == 'rotation' and len(shape.points) < 5:
                        corners = shape._rotation_corners()
                        if len(corners) == 4:
                            shape.points = list(corners)
                            shape.point_labels = [1, 1, 1, 1]
                            handle_pos = shape._rotation_handle_pos()
                            shape.points.append(handle_pos)
                            shape.point_labels.append(1)
                    shapes.append(shape)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Shape Conversion Error",
                    f"Error converting detections to shapes:\n\n{str(e)}\n\n"
                    f"The detection results may be corrupted."
                )
                return

            # In replace mode, confirm before wiping existing annotations
            do_replace = not append
            if do_replace and self.canvas.shapes:
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Replace Existing Annotations?",
                    f"This image already has {len(self.canvas.shapes)} annotation(s).\n\n"
                    "Auto-label will replace them with the new detections.\n"
                    "Enable 'Append' to keep existing annotations instead.",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
                if reply != QtWidgets.QMessageBox.Yes:
                    return

            # Append mode: filter out new shapes that overlap existing ones
            if not do_replace and self.canvas.shapes:
                _iou_thr = self._ai_panel.nms_iou_threshold()
                def _bbox(s):
                    xs = [p.x() for p in s.points]; ys = [p.y() for p in s.points]
                    return min(xs), min(ys), max(xs), max(ys)
                def _iou(a, b):
                    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
                    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
                    if not inter: return 0.0
                    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
                existing_bboxes = [_bbox(es) for es in self.canvas.shapes if es.points]
                shapes = [s for s in shapes if not s.points or
                          all(_iou(_bbox(s), eb) < _iou_thr for eb in existing_bboxes)]

            try:
                self.loadShapes(shapes, replace=do_replace)
                self.setDirty()

                mode = "appended" if not do_replace else "labeled"
                self.statusBar().showMessage(
                    f"✓ Auto-label: {len(shapes)} object(s) {mode}",
                    8000
                )
                print(f"[AUTO-LABEL] {len(shapes)} objects {mode} (conf ≥ {conf_threshold:.0%})")
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Display Error",
                    f"Error displaying auto-labeled shapes:\n\n{str(e)}\n\n"
                    f"The shapes were detected but cannot be displayed."
                )
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            # Catch-all for unexpected errors
            QtWidgets.QMessageBox.critical(
                self,
                "Unexpected Error",
                f"An unexpected error occurred:\n\n"
                f"{type(e).__name__}: {str(e)}\n\n"
                f"Please check the console for detailed error information."
            )
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Batch Auto Label
    # ------------------------------------------------------------------

    def _run_batch_auto_label(self, conf_threshold: float, append: bool = False):
        """Run auto-label on every image in the current folder and save JSON files."""
        import json as _json
        import cv2 as _cv2

        IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        iou_threshold = self._ai_panel.nms_iou_threshold()

        # ── Guard checks ────────────────────────────────────────────────────────
        if not hasattr(self, 'yolo_model') or self.yolo_model is None:
            self._load_pretrained_yolo26()
        if not hasattr(self, 'yolo_model') or self.yolo_model is None:
            return

        folder = getattr(self, 'lastOpenDir', None)
        if not folder or not os.path.isdir(folder):
            QtWidgets.QMessageBox.warning(
                self, "No Folder Open",
                "Please open a folder containing images first.\n\n"
                "Use File → Open Dir to open a folder."
            )
            return

        # ── Find images (recursive) ─────────────────────────────────────────────
        image_paths = sorted(
            os.path.join(root, f)
            for root, _dirs, files in os.walk(folder)
            for f in files
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        if not image_paths:
            QtWidgets.QMessageBox.information(
                self, "No Images Found",
                f"No images found in:\n{folder}\n\n(including subfolders)"
            )
            return

        already_labeled = [
            p for p in image_paths
            if os.path.exists(os.path.splitext(p)[0] + '.json')
        ]

        # ── Load class list ─────────────────────────────────────────────────────
        class_list = None
        if getattr(self, 'dataset_path', None):
            lj = os.path.join(self.dataset_path, ".labels.json")
            if os.path.exists(lj):
                try:
                    with open(lj, "r", encoding="utf-8") as f:
                        class_list = _json.load(f).get("labels")
                except Exception:
                    pass

        # ── Confirmation dialog ─────────────────────────────────────────────────
        detail = (
            f"Folder:  {os.path.basename(folder)}\n\n"
            f"  Images found:         {len(image_paths)}\n"
            f"  Already annotated:    {len(already_labeled)}\n"
            f"  Will be processed:    {len(image_paths)}\n\n"
            f"Confidence threshold:  {conf_threshold:.0%}\n\n"
            "Choose how to handle images that already have annotations:"
        )
        confirm = QtWidgets.QMessageBox(self)
        confirm.setWindowTitle("Batch Auto Label")
        confirm.setText(detail)
        confirm.setIcon(QtWidgets.QMessageBox.Question)
        skip_btn    = confirm.addButton("Skip Existing",  QtWidgets.QMessageBox.ActionRole)
        append_btn  = confirm.addButton("Append",         QtWidgets.QMessageBox.ActionRole)
        replace_btn = confirm.addButton("Replace All",    QtWidgets.QMessageBox.ActionRole)
        cancel_btn  = confirm.addButton("Cancel",         QtWidgets.QMessageBox.RejectRole)
        default_btn = append_btn if append else skip_btn
        confirm.setDefaultButton(default_btn)
        _spacer = QtWidgets.QSpacerItem(520, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        confirm.layout().addItem(_spacer, confirm.layout().rowCount(), 0, 1, confirm.layout().columnCount())
        confirm.exec_()

        clicked = confirm.clickedButton()
        if clicked is cancel_btn or clicked is None:
            return
        skip_existing = (clicked is skip_btn)
        append        = (clicked is append_btn)

        # ── Progress dialog ─────────────────────────────────────────────────────
        progress = QtWidgets.QProgressDialog(
            "Preparing…", "Cancel", 0, len(image_paths), self
        )
        progress.setWindowTitle("Batch Auto Label")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QtWidgets.QApplication.processEvents()
        # resize() must come after show()+processEvents() — QProgressDialog calls
        # adjustSize() internally on show, which overrides setMinimumWidth.
        progress.resize(520, 160)
        QtWidgets.QApplication.processEvents()

        # Build a path→item map for fast real-time file list updates
        _path_to_item = {}
        for _idx in range(self.fileListWidget.count()):
            _it = self.fileListWidget.item(_idx)
            _path_to_item[os.path.abspath(_it.text())] = _it

        def _mark_file_labeled(abs_path):
            it = _path_to_item.get(os.path.abspath(abs_path))
            if it:
                it.setCheckState(Qt.Checked)
                self._set_file_item_color(it, True)

        # ── Process loop ────────────────────────────────────────────────────────
        labeled_count  = 0
        skipped_count  = 0
        empty_count    = 0
        error_msgs     = []
        current_updated = False
        labeled_paths  = []

        for i, img_path in enumerate(image_paths):
            if progress.wasCanceled():
                break

            fname = os.path.basename(img_path)
            json_path = os.path.splitext(img_path)[0] + '.json'

            progress.setLabelText(
                f"[{i + 1} / {len(image_paths)}]  {fname}\n\n"
                f"  Labeled: {labeled_count}   Skipped: {skipped_count}"
                + (f"   Errors: {len(error_msgs)}" if error_msgs else "")
            )
            progress.setValue(i)
            QtWidgets.QApplication.processEvents()

            # Skip if already annotated
            if skip_existing and os.path.exists(json_path):
                skipped_count += 1
                continue

            # Run inference
            try:
                shape_dicts = auto_label_image(
                    self.yolo_model, img_path,
                    class_list=class_list,
                    conf_threshold=conf_threshold,
                )
            except Exception as e:
                error_msgs.append(f"{fname}: {str(e)[:100]}")
                continue

            if not shape_dicts:
                empty_count += 1
                continue

            # Read image dimensions
            try:
                img_bgr = _cv2.imread(img_path)
                if img_bgr is None:
                    raise ValueError("cv2.imread returned None")
                img_h, img_w = img_bgr.shape[:2]
            except Exception as e:
                error_msgs.append(f"{fname}: cannot read dimensions — {e}")
                continue

            # Build labelme JSON
            shapes_json = []
            for sd in shape_dicts:
                shapes_json.append({
                    "label":       sd.get("label", ""),
                    "points":      [[float(x), float(y)] for x, y in sd.get("points", [])],
                    "group_id":    sd.get("group_id"),
                    "description": sd.get("description", ""),
                    "shape_type":  sd.get("shape_type", "polygon"),
                    "flags":       sd.get("flags", {}),
                    "other_data":  sd.get("other_data", {}),
                })

            # Load existing JSON to preserve top-level fields (and shapes when appending)
            existing_data = {}
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        existing_data = _json.load(f)
                except Exception:
                    pass

            if append and existing_data.get("shapes"):
                # Deduplicate: keep new shapes that don't overlap existing ones
                def _bbox_json(pts):
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    return min(xs), min(ys), max(xs), max(ys)
                def _iou_json(a, b):
                    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
                    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
                    if not inter: return 0.0
                    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
                ex_bboxes = [_bbox_json(s["points"]) for s in existing_data["shapes"] if s.get("points")]
                filtered = [s for s in shapes_json if not s["points"] or
                            all(_iou_json(_bbox_json(s["points"]), eb) < iou_threshold for eb in ex_bboxes)]
                final_shapes = existing_data["shapes"] + filtered
            else:
                final_shapes = shapes_json

            json_data = {
                "version":     existing_data.get("version", "5.5.0"),
                "flags":       existing_data.get("flags", {}),
                "shapes":      final_shapes,
                "imagePath":   os.path.basename(img_path),
                "imageData":   None,
                "imageHeight": img_h,
                "imageWidth":  img_w,
            }

            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    _json.dump(json_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                error_msgs.append(f"{fname}: write failed — {e}")
                continue

            labeled_count += 1
            labeled_paths.append(img_path)
            _mark_file_labeled(img_path)

            if self.imagePath and os.path.abspath(img_path) == os.path.abspath(self.imagePath):
                current_updated = True

        progress.setValue(len(image_paths))
        progress.close()

        # ── Reload current image if it was relabeled ────────────────────────────
        if current_updated:
            self.loadFile(self.imagePath)

        # Final pass: sync any remaining items (skipped files, empty detections)
        for idx in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(idx)
            fpath = item.text()
            txt = os.path.splitext(fpath)[0] + '.txt'
            json_f = os.path.splitext(fpath)[0] + '.json'
            has_label = os.path.exists(txt) or os.path.exists(json_f)
            item.setCheckState(Qt.Checked if has_label else Qt.Unchecked)
            self._set_file_item_color(item, has_label)

        # ── Summary ─────────────────────────────────────────────────────────────
        was_canceled = progress.wasCanceled()
        summary_lines = [
            f"{'Batch Auto Label — Canceled' if was_canceled else 'Batch Auto Label Complete'}\n",
            f"  ✓  Labeled:        {labeled_count} image(s)",
            f"  ⏭  Skipped:        {skipped_count} image(s)  (already had annotations)",
            f"  ○  No detections:  {empty_count} image(s)",
        ]
        if error_msgs:
            summary_lines.append(f"  ✗  Errors:         {len(error_msgs)} image(s)")
        summary = "\n".join(summary_lines)

        if error_msgs:
            summary += "\n\nFirst errors:\n" + "\n".join(
                f"  • {e}" for e in error_msgs[:5]
            )
            if len(error_msgs) > 5:
                summary += f"\n  … and {len(error_msgs) - 5} more"

        icon = (QtWidgets.QMessageBox.Warning if error_msgs
                else QtWidgets.QMessageBox.Information)

        if labeled_paths:
            mb = QtWidgets.QMessageBox(icon, "Batch Auto Label", summary, parent=self)
            spacer = QtWidgets.QSpacerItem(520, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
            layout = mb.layout()
            layout.addItem(spacer, layout.rowCount(), 0, 1, layout.columnCount())
            review_btn = mb.addButton("Review Labeled Images", QtWidgets.QMessageBox.ActionRole)
            mb.addButton(QtWidgets.QMessageBox.Ok)
            mb.exec_()
            if mb.clickedButton() is review_btn:
                self._enter_review_mode(labeled_paths)
        else:
            _show_msgbox(icon, "Batch Auto Label", summary, parent=self)

        self.statusBar().showMessage(
            f"Batch Auto Label: {labeled_count} labeled, {skipped_count} skipped"
            + (f", {len(error_msgs)} error(s)" if error_msgs else ""),
            10000,
        )

    # ------------------------------------------------------------------
    # Batch Magic Label
    # ------------------------------------------------------------------

    def _run_batch_magic_label(self, prompt: str, label_names: str,
                                box_threshold: float, text_threshold: float,
                                use_sam: bool = False, append: bool = False):
        """Run Magic Label (Grounding DINO + optional SAM2) on every image in the folder."""
        import json as _json
        import cv2 as _cv2

        IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        iou_threshold = self._ai_panel.nms_iou_threshold()

        # ── Guard: prompt required ───────────────────────────────────────────────
        prompt = prompt.strip()
        if not prompt:
            QtWidgets.QMessageBox.warning(
                self, "No Prompt",
                "Please enter the objects to detect in the 'Objects to detect' field\n"
                "before running batch Magic Label."
            )
            return

        # ── Guard: folder open ───────────────────────────────────────────────────
        folder = getattr(self, 'lastOpenDir', None)
        if not folder or not os.path.isdir(folder):
            QtWidgets.QMessageBox.warning(
                self, "No Folder Open",
                "Please open a folder containing images first.\n\n"
                "Use File → Open Dir to open a folder."
            )
            return

        # ── Find images (recursive) ─────────────────────────────────────────────
        image_paths = sorted(
            os.path.join(root, f)
            for root, _dirs, files in os.walk(folder)
            for f in files
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        if not image_paths:
            QtWidgets.QMessageBox.information(
                self, "No Images Found",
                f"No images found in:\n{folder}\n\n(including subfolders)"
            )
            return

        already_labeled = [
            p for p in image_paths
            if os.path.exists(os.path.splitext(p)[0] + '.json')
        ]

        # ── Build label name mapping ─────────────────────────────────────────────
        prompt_parts = [p.strip() for p in prompt.replace(".", ",").split(",") if p.strip()]
        name_parts   = [n.strip() for n in label_names.split(",") if n.strip()]
        label_map = {
            prompt_parts[i]: (name_parts[i] if i < len(name_parts) else prompt_parts[i])
            for i in range(len(prompt_parts))
        }

        # SAM2 only applies in segmentation mode
        effective_use_sam = use_sam and (self.task_type == "segmentation")

        # ── Confirmation dialog ─────────────────────────────────────────────────
        sam_line = ("SAM2 masks:  enabled (segmentation polygons)" if effective_use_sam
                    else "SAM2 masks:  disabled (bounding boxes only)")
        detail = (
            f"Folder:  {os.path.basename(folder)}\n\n"
            f"  Images found:         {len(image_paths)}\n"
            f"  Already annotated:    {len(already_labeled)}\n\n"
            f"Prompt:  \"{prompt}\"\n"
            f"Box threshold:  {box_threshold:.0%}   "
            f"Text threshold:  {text_threshold:.0%}\n"
            f"{sam_line}\n\n"
            "Choose how to handle images that already have annotations:"
        )
        confirm = QtWidgets.QMessageBox(self)
        confirm.setWindowTitle("Batch Magic Label")
        confirm.setText(detail)
        confirm.setIcon(QtWidgets.QMessageBox.Question)
        skip_btn    = confirm.addButton("Skip Existing",  QtWidgets.QMessageBox.ActionRole)
        append_btn  = confirm.addButton("Append",         QtWidgets.QMessageBox.ActionRole)
        replace_btn = confirm.addButton("Replace All",    QtWidgets.QMessageBox.ActionRole)
        cancel_btn  = confirm.addButton("Cancel",         QtWidgets.QMessageBox.RejectRole)
        default_btn = append_btn if append else skip_btn
        confirm.setDefaultButton(default_btn)
        _spacer = QtWidgets.QSpacerItem(520, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        confirm.layout().addItem(_spacer, confirm.layout().rowCount(), 0, 1, confirm.layout().columnCount())
        confirm.exec_()

        clicked = confirm.clickedButton()
        if clicked is cancel_btn or clicked is None:
            return
        skip_existing = (clicked is skip_btn)
        append        = (clicked is append_btn)

        # ── Import magic_label (may trigger model download on first run) ─────────
        init_msg = ("Loading Grounding DINO + SAM2 models…\n(first run may take a moment)"
                    if effective_use_sam else
                    "Loading Grounding DINO model…\n(first run may take a moment)")
        progress_init = QtWidgets.QProgressDialog(init_msg, "", 0, 0, self)
        progress_init.setWindowTitle("Batch Magic Label")
        progress_init.setWindowModality(QtCore.Qt.WindowModal)
        progress_init.setCancelButton(None)
        progress_init.show()
        QtWidgets.QApplication.processEvents()
        progress_init.resize(520, 120)
        QtWidgets.QApplication.processEvents()
        try:
            from labelme.magic_label import magic_label_image as _magic_label_image
        except ImportError as e:
            progress_init.close()
            QtWidgets.QMessageBox.critical(
                self, "Missing Dependency",
                f"Magic Label requires the 'transformers' package:\n\n{e}\n\n"
                "Install it with:  pip install transformers"
            )
            return
        progress_init.close()

        # ── Progress dialog ─────────────────────────────────────────────────────
        _cancelled = [False]
        progress = QtWidgets.QProgressDialog(
            "Preparing…", "Cancel", 0, len(image_paths), self
        )
        progress.setWindowTitle("Batch Magic Label")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.canceled.connect(lambda: _cancelled.__setitem__(0, True))
        progress.show()
        QtWidgets.QApplication.processEvents()
        progress.resize(520, 160)
        QtWidgets.QApplication.processEvents()

        # Build a path→item map for fast real-time file list updates
        _path_to_item = {}
        for _idx in range(self.fileListWidget.count()):
            _it = self.fileListWidget.item(_idx)
            _path_to_item[os.path.abspath(_it.text())] = _it

        def _mark_file_labeled(abs_path):
            it = _path_to_item.get(os.path.abspath(abs_path))
            if it:
                it.setCheckState(Qt.Checked)
                self._set_file_item_color(it, True)

        # ── Process loop ────────────────────────────────────────────────────────
        labeled_count   = 0
        skipped_count   = 0
        empty_count     = 0
        error_msgs      = []
        current_updated = False
        labeled_paths   = []

        for i, img_path in enumerate(image_paths):
            if _cancelled[0]:
                break

            fname     = os.path.basename(img_path)
            json_path = os.path.splitext(img_path)[0] + '.json'

            progress.setLabelText(
                f"[{i + 1} / {len(image_paths)}]  {fname}\n\n"
                f"  Labeled: {labeled_count}   Skipped: {skipped_count}"
                + (f"   Errors: {len(error_msgs)}" if error_msgs else "")
            )
            progress.setValue(i)
            QtWidgets.QApplication.processEvents()

            if _cancelled[0]:
                break

            if skip_existing and os.path.exists(json_path):
                skipped_count += 1
                continue

            # Run Grounding DINO (SAM2 always off in batch)
            try:
                shape_dicts = _magic_label_image(
                    img_path, prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    use_sam=effective_use_sam,
                    stop_callback=lambda: _cancelled[0],
                )
            except Exception as e:
                if _cancelled[0]:
                    break
                error_msgs.append(f"{fname}: {str(e)[:100]}")
                continue

            if _cancelled[0]:
                break

            # Apply label name mapping
            if name_parts:
                for sd in shape_dicts:
                    detected = sd.get("label", "")
                    mapped = label_map.get(detected)
                    if mapped is None:
                        for phrase, name in label_map.items():
                            if (detected.lower() in phrase.lower()
                                    or phrase.lower() in detected.lower()):
                                mapped = name
                                break
                    if mapped:
                        sd["label"] = mapped

            # Convert shapes to the correct format for the current task type
            for sd in shape_dicts:
                if sd.get("shape_type") == "rectangle":
                    pts = sd.get("points", [])
                    if self.task_type == "segmentation":
                        # Rectangle → 4-corner polygon so it survives the load filter
                        if len(pts) == 2:
                            (x1, y1), (x2, y2) = pts[0], pts[1]
                            sd["points"] = [
                                [min(x1, x2), min(y1, y2)],
                                [max(x1, x2), min(y1, y2)],
                                [max(x1, x2), max(y1, y2)],
                                [min(x1, x2), max(y1, y2)],
                            ]
                        sd["shape_type"] = "polygon"
                    elif self.task_type == "obb":
                        sd["shape_type"] = "rotation"
                        sd.setdefault("other_data", {})["rotation"] = 0.0

            if not shape_dicts:
                empty_count += 1
                continue

            # Read image dimensions
            try:
                img_bgr = _cv2.imread(img_path)
                if img_bgr is None:
                    raise ValueError("cv2.imread returned None")
                img_h, img_w = img_bgr.shape[:2]
            except Exception as e:
                error_msgs.append(f"{fname}: cannot read dimensions — {e}")
                continue

            # Build labelme JSON
            shapes_json = [
                {
                    "label":       sd.get("label", ""),
                    "points":      [[float(x), float(y)] for x, y in sd.get("points", [])],
                    "group_id":    sd.get("group_id"),
                    "description": sd.get("description", ""),
                    "shape_type":  sd.get("shape_type", "polygon"),
                    "flags":       sd.get("flags", {}),
                    "other_data":  sd.get("other_data", {}),
                }
                for sd in shape_dicts
            ]

            existing_data = {}
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        existing_data = _json.load(f)
                except Exception:
                    pass

            if append and existing_data.get("shapes"):
                def _bbox_json(pts):
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    return min(xs), min(ys), max(xs), max(ys)
                def _iou_json(a, b):
                    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
                    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
                    if not inter: return 0.0
                    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
                ex_bboxes = [_bbox_json(s["points"]) for s in existing_data["shapes"] if s.get("points")]
                filtered = [s for s in shapes_json if not s["points"] or
                            all(_iou_json(_bbox_json(s["points"]), eb) < iou_threshold for eb in ex_bboxes)]
                final_shapes = existing_data["shapes"] + filtered
            else:
                final_shapes = shapes_json

            json_data = {
                "version":     existing_data.get("version", "5.5.0"),
                "flags":       existing_data.get("flags", {}),
                "shapes":      final_shapes,
                "imagePath":   os.path.basename(img_path),
                "imageData":   None,
                "imageHeight": img_h,
                "imageWidth":  img_w,
            }

            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    _json.dump(json_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                error_msgs.append(f"{fname}: write failed — {e}")
                continue

            labeled_count += 1
            labeled_paths.append(img_path)
            _mark_file_labeled(img_path)

            if self.imagePath and os.path.abspath(img_path) == os.path.abspath(self.imagePath):
                current_updated = True

        progress.setValue(len(image_paths))
        progress.close()

        # ── Release model memory so the app stays responsive ─────────────────────
        try:
            from labelme.magic_label import clear_model_cache as _clear_magic_cache
            _clear_magic_cache()
            import gc as _gc
            _gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass

        # ── Reload current image if relabeled ────────────────────────────────────
        if current_updated:
            self.loadFile(self.imagePath)

        for idx in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(idx)
            fpath = item.text()
            txt  = os.path.splitext(fpath)[0] + '.txt'
            jf   = os.path.splitext(fpath)[0] + '.json'
            has_label = os.path.exists(txt) or os.path.exists(jf)
            item.setCheckState(Qt.Checked if has_label else Qt.Unchecked)
            self._set_file_item_color(item, has_label)

        # ── Summary ─────────────────────────────────────────────────────────────
        was_canceled = _cancelled[0]
        summary_lines = [
            f"{'Batch Magic Label — Canceled' if was_canceled else 'Batch Magic Label Complete'}\n",
            f"  ✓  Labeled:        {labeled_count} image(s)",
            f"  ⏭  Skipped:        {skipped_count} image(s)  (already had annotations)",
            f"  ○  No detections:  {empty_count} image(s)",
        ]
        if error_msgs:
            summary_lines.append(f"  ✗  Errors:         {len(error_msgs)} image(s)")
        summary = "\n".join(summary_lines)
        if error_msgs:
            summary += "\n\nFirst errors:\n" + "\n".join(
                f"  • {e}" for e in error_msgs[:5]
            )
            if len(error_msgs) > 5:
                summary += f"\n  … and {len(error_msgs) - 5} more"

        icon = (QtWidgets.QMessageBox.Warning if error_msgs
                else QtWidgets.QMessageBox.Information)

        if labeled_paths:
            mb = QtWidgets.QMessageBox(icon, "Batch Magic Label", summary, parent=self)
            spacer = QtWidgets.QSpacerItem(520, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
            layout = mb.layout()
            layout.addItem(spacer, layout.rowCount(), 0, 1, layout.columnCount())
            review_btn = mb.addButton("Review Labeled Images", QtWidgets.QMessageBox.ActionRole)
            mb.addButton(QtWidgets.QMessageBox.Ok)
            mb.exec_()
            if mb.clickedButton() is review_btn:
                self._enter_review_mode(labeled_paths)
        else:
            _show_msgbox(icon, "Batch Magic Label", summary, parent=self)

        self.statusBar().showMessage(
            f"Batch Magic Label: {labeled_count} labeled, {skipped_count} skipped"
            + (f", {len(error_msgs)} error(s)" if error_msgs else ""),
            10000,
        )

    # ------------------------------------------------------------------
    # Magic Label  (Grounding DINO + SAM2)
    # ------------------------------------------------------------------

    def magic_label_slot(self):
        """Open the floating AI panel in Magic Label mode."""
        if not self.imagePath or not os.path.exists(str(self.imagePath)):
            QtWidgets.QMessageBox.warning(self, "No Image",
                "Please open an image before running Magic Label.")
            return
        # Pre-fill prompt with accumulated labels from this session
        accumulated = getattr(self, "_magic_label_accumulated_prompts", [])
        prefill = ", ".join(accumulated) if accumulated else getattr(self, "_magic_label_last_prompt", "")
        self._ai_panel.set_prompt(prefill)
        # Configure SAM2 checkbox based on task mode
        self._ai_panel.configure_for_task(getattr(self, "task_type", "segmentation"))
        # Default append checkbox
        self._ai_panel.set_append_default(bool(self.canvas.shapes))
        self._ai_panel.show_mode("magic")

    def _run_magic_label_from_panel(self, prompt: str, label_names: str, box_threshold: float,
                                     text_threshold: float, use_sam: bool, append: bool):
        """Run Grounding DINO + SAM2 — called by the panel's Run button."""
        prompt = prompt.strip()
        if not prompt:
            QtWidgets.QMessageBox.warning(self, "Empty Prompt",
                "Please enter at least one object name.")
            return
        if not self.imagePath or not os.path.exists(str(self.imagePath)):
            return

        do_replace = not append

        # Warn before replacing existing annotations
        if do_replace and self.canvas.shapes:
            reply = QtWidgets.QMessageBox.question(
                self, "Replace existing annotations?",
                f"This will replace {len(self.canvas.shapes)} existing annotation(s).\n\nContinue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        progress = _MagicProgressDialog(self)
        progress.show()
        QtWidgets.QApplication.processEvents()

        # Build prompt → label name mapping (position-matched, comma-separated)
        prompt_parts = [p.strip() for p in prompt.replace(".", ",").split(",") if p.strip()]
        name_parts   = [n.strip() for n in label_names.split(",") if n.strip()]
        # Map each prompt phrase to its label name; fall back to the prompt phrase itself
        _label_map = {
            prompt_parts[i]: (name_parts[i] if i < len(name_parts) else prompt_parts[i])
            for i in range(len(prompt_parts))
        }

        # bbox and OBB modes use rectangles — never SAM2 polygons
        # (OBB rectangles are converted to rotation shapes after detection)
        effective_use_sam = use_sam and (self.task_type == "segmentation")
        self._magic_label_worker = MagicLabelWorker(
            self.imagePath, prompt, box_threshold, text_threshold, effective_use_sam
        )
        progress.canceled.connect(self._magic_label_worker.cancel)

        def on_status(msg):
            progress.setLabelText(msg)
            QtWidgets.QApplication.processEvents()

        def on_finished(shape_dicts):
            progress.close()
            if not shape_dicts:
                QtWidgets.QMessageBox.information(self, "Magic Label",
                    "No objects detected.\n\n"
                    "Try lowering the threshold or using different object names.")
                return

            # Apply label name mapping if the user provided label names
            if name_parts:
                for sd in shape_dicts:
                    detected = sd.get("label", "")
                    # Exact match first, then substring match
                    mapped = _label_map.get(detected)
                    if mapped is None:
                        for prompt_phrase, label_name in _label_map.items():
                            if detected.lower() in prompt_phrase.lower() or prompt_phrase.lower() in detected.lower():
                                mapped = label_name
                                break
                    if mapped:
                        sd["label"] = mapped

            shapes = []
            for sd in shape_dicts:
                shape_type = sd.get("shape_type", "polygon")
                other_data = dict(sd.get("other_data") or {})

                # In OBB mode Grounding-DINO always returns axis-aligned rectangles.
                # Convert them to rotation shapes (angle=0) so they behave like
                # shapes drawn with the OBB tool and can be rotated/resized normally.
                if shape_type == "rectangle" and self.task_type == "obb":
                    shape_type = "rotation"
                    other_data.setdefault("rotation", 0.0)

                shape = Shape(
                    label=sd.get("label", ""),
                    shape_type=shape_type,
                    group_id=sd.get("group_id"),
                    description=sd.get("description", ""),
                    mask=sd.get("mask"),
                )
                for pt in sd.get("points", []):
                    shape.addPoint(QtCore.QPointF(pt[0], pt[1]))
                shape.flags = sd.get("flags", {})
                shape.other_data = other_data
                if sd.get("confidence") is not None:
                    shape.confidence = float(sd["confidence"])
                if shape.shape_type == "polygon":
                    shape.close()

                # Ensure rotation shapes are in 5-point format (4 corners + handle)
                # so they can be resized and rotated via the normal OBB vertex handles.
                if shape.shape_type == "rotation" and len(shape.points) < 5:
                    corners = shape._rotation_corners()
                    if len(corners) == 4:
                        shape.points = list(corners)
                        shape.point_labels = [1, 1, 1, 1]
                        handle_pos = shape._rotation_handle_pos()
                        shape.points.append(handle_pos)
                        shape.point_labels.append(1)

                shapes.append(shape)

            # Append mode: skip new shapes that overlap existing ones
            if not do_replace and self.canvas.shapes:
                _iou_thr = self._ai_panel.nms_iou_threshold()
                def _bbox(s):
                    xs = [p.x() for p in s.points]; ys = [p.y() for p in s.points]
                    return min(xs), min(ys), max(xs), max(ys)
                def _iou(a, b):
                    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
                    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
                    if not inter: return 0.0
                    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
                existing_bboxes = [_bbox(es) for es in self.canvas.shapes if es.points]
                shapes = [s for s in shapes if not s.points or
                          all(_iou(_bbox(s), eb) < _iou_thr for eb in existing_bboxes)]

            self.loadShapes(shapes, replace=do_replace)
            self.setDirty()
            mode = "replaced" if do_replace else "appended"
            self.statusBar().showMessage(f"Magic Label: {len(shapes)} object(s) {mode}", 5000)

            # Save state and accumulate prompts
            self._magic_label_last_prompt = prompt
            self._magic_label_last_box_threshold = box_threshold
            self._magic_label_last_text_threshold = text_threshold
            new_labels = [l.strip() for l in prompt.replace(".", ",").split(",") if l.strip()]
            existing = getattr(self, "_magic_label_accumulated_prompts", [])
            for lbl in new_labels:
                if lbl not in existing:
                    existing.append(lbl)
            self._magic_label_accumulated_prompts = existing
            # Do NOT overwrite panel prompt — keep whatever the user typed

        def on_error(msg):
            progress.close()
            QtWidgets.QMessageBox.critical(self, "Magic Label Error", msg)

        self._magic_label_worker.status_signal.connect(on_status)
        self._magic_label_worker.finished_signal.connect(on_finished)
        self._magic_label_worker.error_signal.connect(on_error)
        self._magic_label_worker.start()

    def clear_magic_label_cache_slot(self):
        """Free cached Grounding DINO and SAM2 models from memory."""
        try:
            from labelme.magic_label import clear_model_cache
            clear_model_cache()
            self.statusBar().showMessage("Magic Label model cache cleared.", 4000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Clear Cache", f"Could not clear cache:\n{e}")

    # ------------------------------------------------------------------
    # Auto-save timer
    # ------------------------------------------------------------------

    def _autosave_tick(self):
        """Save quietly every 2 minutes if there are unsaved changes."""
        if self.dirty and self.filename and os.path.isfile(str(self.filename)):
            try:
                self.saveFile()
                self.statusBar().showMessage("Auto-saved.", 2000)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Keyboard shortcut: E — toggle Verified Empty
    # ------------------------------------------------------------------

    def toggleVerifiedEmpty(self):
        """Toggle the 'No Objects (Verified Empty)' checkbox via keyboard."""
        if hasattr(self, 'verified_empty_checkbox') and self.imagePath:
            new_state = not self.verified_empty_checkbox.isChecked()
            self.verified_empty_checkbox.setChecked(new_state)

    # ------------------------------------------------------------------
    # Quick review mode
    # ------------------------------------------------------------------

    def _enter_review_mode(self, image_paths: list):
        """Enter review mode: navigate through image_paths one by one."""
        if not image_paths:
            return
        self._review_queue = list(image_paths)
        self._review_index = 0
        self._in_review_mode = True
        self._review_load_current()

    def _review_load_current(self):
        if not self._in_review_mode or self._review_index >= len(self._review_queue):
            self._exit_review_mode()
            return
        img_path = self._review_queue[self._review_index]
        self.loadFile(img_path)
        n = len(self._review_queue)
        i = self._review_index + 1
        self.statusBar().showMessage(
            f"[REVIEW {i}/{n}]  Space = keep & next  |  Del = clear & next  |  Esc = exit review",
            0  # persistent
        )

    def _review_approve(self):
        """Keep annotations and advance to next image."""
        if self.dirty:
            self.saveFile()
        self._review_index += 1
        self._review_load_current()

    def _review_delete_and_next(self):
        """Remove all annotations on current image, save, and advance."""
        if self._in_review_mode and self.imagePath:
            self.loadShapes([], replace=True)
            self.setDirty()
            self.saveFile()
            self._review_index += 1
            self._review_load_current()

    def _exit_review_mode(self):
        done = self._review_index
        self._in_review_mode = False
        self._review_queue = []
        self._review_index = 0
        if done > 0:
            self.statusBar().showMessage(f"Review complete — {done} image(s) reviewed.", 6000)
        else:
            self.statusBar().showMessage("Review mode exited.", 3000)

    def keyPressEvent(self, event):
        if self._in_review_mode:
            key = event.key()
            if key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Right):
                self._review_approve()
                return
            if key == Qt.Key_Delete:
                self._review_delete_and_next()
                return
            if key == Qt.Key_Escape:
                self._exit_review_mode()
                return
        # E — verify as empty (outside drawing mode to avoid conflicts)
        if event.key() == Qt.Key_E and not event.modifiers():
            if not self.canvas.drawing():
                self.toggleVerifiedEmpty()
                return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Dataset statistics panel
    # ------------------------------------------------------------------

    def show_stats_smart(self):
        """Show rich YOLO stats if an export exists, otherwise warn the user."""
        dp = getattr(self, 'dataset_path', None)
        labels_json = os.path.join(dp, '.labels.json') if dp else None
        labels_dir  = os.path.join(dp, 'labels')       if dp else None

        if dp and os.path.isdir(dp) and labels_json and os.path.exists(labels_json) \
                and labels_dir and os.path.isdir(labels_dir):
            show_yolo_stats(self, dp)
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Dataset Stats",
                "No exported dataset found.\n\n"
                "Export your annotations first via\n"
                "Tools → Export to YOLO, then open stats."
            )

    def show_dataset_stats(self):
        """Scan the current project folder and show per-class annotation counts."""
        folder = getattr(self, 'lastOpenDir', None)
        if not folder or not os.path.isdir(folder):
            QtWidgets.QMessageBox.information(
                self, "Dataset Stats",
                "No folder is open.\nOpen a project folder first (File → Open Dir)."
            )
            return

        import json as _json
        from collections import Counter

        class_counts: Counter = Counter()
        total_images = 0
        labeled_images = 0
        empty_images = 0

        for root, _dirs, files in os.walk(folder):
            for fname in files:
                if not fname.endswith('.json') or fname == '.labels.json':
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = _json.load(f)
                except Exception:
                    continue
                # Only count if a matching image exists
                img_name = data.get('imagePath', '')
                img_path = os.path.join(root, img_name)
                if not os.path.exists(img_path):
                    img_path = os.path.join(folder, img_name)
                if not os.path.exists(img_path):
                    continue
                total_images += 1
                shapes = data.get('shapes', [])
                if data.get('verified_empty'):
                    empty_images += 1
                    labeled_images += 1
                elif shapes:
                    labeled_images += 1
                    for s in shapes:
                        lbl = s.get('label', '')
                        if lbl:
                            class_counts[lbl] += 1

        # Build text
        lines = [
            f"Project folder:  {os.path.basename(folder)}\n",
            f"  Images with annotations:  {labeled_images}",
            f"  Verified empty:           {empty_images}",
            f"  Unlabeled:                {total_images - labeled_images}\n",
        ]
        if class_counts:
            lines.append("Class distribution:")
            for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {cls:<30s}  {cnt:>5d} annotation(s)")
        else:
            lines.append("No annotations found.")

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Dataset Statistics")
        dlg.setMinimumWidth(480)
        layout = QtWidgets.QVBoxLayout(dlg)
        text_widget = QtWidgets.QPlainTextEdit("\n".join(lines))
        text_widget.setReadOnly(True)
        text_widget.setFont(QtGui.QFont("Courier", 10))
        layout.addWidget(text_widget)
        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.exec_()

    # ------------------------------------------------------------------
    # Magic Label model warm-up
    # ------------------------------------------------------------------

    def warmup_magic_label_models(self):
        """Pre-load Grounding DINO (and SAM2 if checked) into memory."""
        use_sam = (hasattr(self._ai_panel, '_sam2_check')
                   and self._ai_panel._sam2_check.isChecked()
                   and self.task_type == "segmentation")

        msg = ("Loading Grounding DINO + SAM2…" if use_sam
               else "Loading Grounding DINO…")
        prog = QtWidgets.QProgressDialog(msg, "", 0, 0, self)
        prog.setWindowTitle("Pre-loading Models")
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.setCancelButton(None)
        prog.show()
        prog.resize(420, 110)
        QtWidgets.QApplication.processEvents()

        try:
            from labelme.magic_label import _load_grounding_dino, _load_sam2
            _load_grounding_dino(progress_callback=lambda m: self.statusBar().showMessage(m, 2000))
            if use_sam:
                _load_sam2(progress_callback=lambda m: self.statusBar().showMessage(m, 2000))
        except Exception as e:
            prog.close()
            QtWidgets.QMessageBox.warning(self, "Model Load Error", str(e))
            return
        prog.close()
        self.statusBar().showMessage("Models loaded and ready.", 4000)

    def show_training_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "YOLO Training Error", msg)