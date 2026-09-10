import os
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QThread, pyqtSignal, QBuffer, QIODevice
from labelme.utils import fit_dialog

THUMB_SIZE = 200
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


class ThumbnailWorker(QThread):
    """
    Two-phase worker:
      1. Scans the directory (fast) → emits scan_done with all paths immediately.
      2. Loads & scales each image in the background → emits thumbnail_ready per image.
    """
    scan_done = pyqtSignal(list, dict)        # (sorted image paths, stats dict)
    thumbnail_ready = pyqtSignal(int, bytes)  # (index in paths list, PNG bytes)

    def __init__(self, dataset_path, split):
        super().__init__()
        self.dataset_path = dataset_path
        self.split = split
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        # ── Phase 1: scan filenames only (no image IO, nearly instant) ──────────
        image_dir = os.path.join(self.dataset_path, "images", self.split)
        images = []
        if os.path.isdir(image_dir):
            for fname in os.listdir(image_dir):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                    images.append(os.path.join(image_dir, fname))
        images = sorted(images)

        train_dir = os.path.join(self.dataset_path, "images", "train")
        val_dir   = os.path.join(self.dataset_path, "images", "val")
        train_count = (
            sum(1 for f in os.listdir(train_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
            if os.path.isdir(train_dir) else 0
        )
        val_count = (
            sum(1 for f in os.listdir(val_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
            if os.path.isdir(val_dir) else 0
        )

        self.scan_done.emit(images, {
            'train_count': train_count,
            'val_count': val_count,
            'current_count': len(images),
        })

        if self._cancelled:
            return

        # ── Phase 2: load thumbnails one-by-one, stream to UI ───────────────────
        for idx, img_path in enumerate(images):
            if self._cancelled:
                break
            try:
                qimg = QtGui.QImage(img_path)
                if qimg.isNull():
                    continue
                scaled = qimg.scaled(
                    THUMB_SIZE, THUMB_SIZE,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                buf = QBuffer()
                buf.open(QIODevice.WriteOnly)
                scaled.save(buf, "PNG")
                buf.close()
                self.thumbnail_ready.emit(idx, bytes(buf.data()))
            except Exception:
                pass


class ExploreYoloDialog(QtWidgets.QDialog):
    def __init__(self, parent, dataset_path, image_loader_callback, progress_dialog=None):
        super().__init__(parent)
        self.setWindowTitle("Explore YOLO Dataset")
        fit_dialog(self, 1200, 800)
        self.image_loader_callback = image_loader_callback
        self.dataset_path = dataset_path
        self.worker = None
        self.images = []
        self._placeholder = self._make_placeholder()

        # ── Layout ───────────────────────────────────────────────────────────────
        layout = QtWidgets.QVBoxLayout(self)

        top_bar = QtWidgets.QHBoxLayout()
        self.split_selector = QtWidgets.QComboBox()
        self.split_selector.addItems(["train", "val"])
        self.split_selector.currentTextChanged.connect(self.on_split_changed)
        top_bar.addWidget(self.split_selector)

        self.loading_label = QtWidgets.QLabel("")
        self.loading_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        top_bar.addWidget(self.loading_label)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setViewMode(QtWidgets.QListView.IconMode)
        self.list_widget.setIconSize(QtCore.QSize(THUMB_SIZE, THUMB_SIZE))
        self.list_widget.setResizeMode(QtWidgets.QListWidget.Adjust)
        self.list_widget.setMovement(QtWidgets.QListView.Static)
        self.list_widget.setSpacing(16)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.setGridSize(QtCore.QSize(230, 250))
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout.addWidget(self.list_widget)

        self.count_label = QtWidgets.QLabel()
        layout.addWidget(self.count_label)

        # Close the pre-open progress dialog immediately — we no longer block
        if progress_dialog:
            progress_dialog.close()

        self._start_loading(self.split_selector.currentText())

    # ── Helpers ──────────────────────────────────────────────────────────────────

    def _make_placeholder(self):
        img = QtGui.QImage(THUMB_SIZE, THUMB_SIZE, QtGui.QImage.Format_RGB888)
        img.fill(QtGui.QColor(55, 55, 55))
        return QtGui.QIcon(QtGui.QPixmap.fromImage(img))

    def _stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(500)
        self.worker = None

    # ── Loading pipeline ─────────────────────────────────────────────────────────

    def _start_loading(self, split):
        self._stop_worker()
        self.list_widget.clear()
        self.images = []
        self.loading_label.setText("Scanning…")

        self.worker = ThumbnailWorker(self.dataset_path, split)
        self.worker.scan_done.connect(self._on_scan_done)
        self.worker.thumbnail_ready.connect(self._on_thumbnail_ready)
        self.worker.start()

    def _on_scan_done(self, images, stats):
        """
        Called as soon as the file list is known.
        Instantly populates every item with a placeholder — no image IO on the main thread.
        """
        self.images = images
        n = len(images)

        for img_path in images:
            item = QtWidgets.QListWidgetItem(os.path.basename(img_path))
            item.setTextAlignment(QtCore.Qt.AlignHCenter)
            item.setIcon(self._placeholder)
            item.setData(QtCore.Qt.UserRole, img_path)
            self.list_widget.addItem(item)

        split = self.split_selector.currentText()
        train_count = stats.get('train_count', 0)
        val_count   = stats.get('val_count', 0)
        self.count_label.setText(
            f"Showing {n} images in '{split}' | "
            f"Train: {train_count} | Val: {val_count} | Total: {train_count + val_count}"
        )
        self.loading_label.setText(f"Loading thumbnails… 0 / {n}")

    def _on_thumbnail_ready(self, idx, png_bytes):
        """Replace a placeholder with the real thumbnail as it arrives."""
        item = self.list_widget.item(idx)
        if item is None:
            return
        px = QtGui.QPixmap()
        px.loadFromData(png_bytes, "PNG")
        if not px.isNull():
            item.setIcon(QtGui.QIcon(px))

        done = idx + 1
        n    = len(self.images)
        self.loading_label.setText("" if done >= n else f"Loading thumbnails… {done} / {n}")

    # ── Slots ─────────────────────────────────────────────────────────────────────

    def on_split_changed(self, split):
        self._start_loading(split)

    def closeEvent(self, event):
        self._stop_worker()
        super().closeEvent(event)

    def on_item_double_clicked(self, item):
        img_path = item.data(QtCore.Qt.UserRole)
        if not img_path:
            return

        # set_explore_file_list populates the file list and calls loadFile internally.
        # loadFile now also loads the YOLO labels via _load_explore_yolo_labels, so
        # there is no need to duplicate the label-loading logic here.
        if hasattr(self.parent(), "set_explore_file_list"):
            self.parent().set_explore_file_list(self.images, img_path)
        elif self.image_loader_callback:
            self.image_loader_callback(img_path)

        self.accept()
