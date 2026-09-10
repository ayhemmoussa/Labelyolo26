import os
import json
import glob
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from labelme.utils import fit_dialog


def make_stat_box(number, label, color1, color2):
    box = QtWidgets.QFrame()
    box.setFrameShape(QtWidgets.QFrame.StyledPanel)
    box.setStyleSheet(f"""
        QFrame {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {color1}, stop:1 {color2});
            border-radius: 28px;
            border: none;
            min-width: 140px;
            min-height: 80px;
        }}
    """)
    layout = QtWidgets.QVBoxLayout(box)
    layout.setContentsMargins(0, 18, 0, 18)
    layout.setSpacing(0)
    num_label = QtWidgets.QLabel(str(number))
    num_label.setAlignment(QtCore.Qt.AlignCenter)
    num_label.setStyleSheet("""
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 1px;
        font-family: Arial Rounded MT Bold, Arial, sans-serif;
    """)
    text_label = QtWidgets.QLabel(label)
    text_label.setAlignment(QtCore.Qt.AlignCenter)
    text_label.setStyleSheet("""
        font-size: 15px;
        color: #cccccc;
        font-weight: 600;
        font-family: Arial Rounded MT Bold, Arial, sans-serif;
        margin-top: 6px;
    """)
    layout.addWidget(num_label)
    layout.addWidget(text_label)
    return box


def show_yolo_stats(parent, dataset_path):
    # ── Load data ────────────────────────────────────────────────────────────────
    labels_path = os.path.join(dataset_path, ".labels.json")
    if not os.path.exists(labels_path):
        QtWidgets.QMessageBox.warning(parent, "Stats", "No .labels.json found in dataset path.")
        return
    with open(labels_path, "r", encoding="utf-8") as f:
        label_names = json.load(f).get("labels", [])

    train_labels = glob.glob(os.path.join(dataset_path, "labels", "train", "*.txt"))
    val_labels   = glob.glob(os.path.join(dataset_path, "labels", "val",   "*.txt"))
    all_labels   = train_labels + val_labels

    train_images = glob.glob(os.path.join(dataset_path, "images", "train", "*"))
    val_images   = glob.glob(os.path.join(dataset_path, "images", "val",   "*"))
    num_train    = len(train_images)
    num_val      = len(val_images)
    num_images   = num_train + num_val

    label_counts     = [0] * len(label_names)
    total_annotations = 0
    for label_file in all_labels:
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    idx = int(parts[0])
                    if 0 <= idx < len(label_counts):
                        label_counts[idx] += 1
                        total_annotations += 1

    # ── Screen-relative dialog size ──────────────────────────────────────────────
    screen = QApplication.primaryScreen().availableGeometry()
    sw, sh = screen.width(), screen.height()
    dlg_w  = min(1400, int(sw * 0.92))
    dlg_h  = min(900,  int(sh * 0.90))

    # ── Dialog ───────────────────────────────────────────────────────────────────
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("YOLO Dataset Statistics")
    fit_dialog(dialog, dlg_w, dlg_h)
    dialog.setMinimumSize(600, 400)

    outer_layout = QtWidgets.QVBoxLayout(dialog)
    outer_layout.setContentsMargins(0, 0, 0, 0)
    outer_layout.setSpacing(0)

    # Scroll area wraps everything so nothing is ever clipped
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    outer_layout.addWidget(scroll)

    container = QtWidgets.QWidget()
    scroll.setWidget(container)
    main_layout = QtWidgets.QVBoxLayout(container)
    main_layout.setContentsMargins(20, 14, 20, 20)
    main_layout.setSpacing(18)

    # ── Stat boxes ───────────────────────────────────────────────────────────────
    color = ("#2a5298", "#1e3c72")
    stat_boxes = [
        make_stat_box(num_images,                   "Images",      *color),
        make_stat_box(total_annotations,             "Annotations", *color),
        make_stat_box(len(label_names),              "Labels",      *color),
        make_stat_box(f"{num_train} / {num_val}",   "Train / Val", *color),
    ]
    stat_row = QtWidgets.QHBoxLayout()
    stat_row.setSpacing(18)
    stat_row.addStretch()
    for box in stat_boxes:
        stat_row.addWidget(box, alignment=QtCore.Qt.AlignVCenter)
    stat_row.addStretch()
    main_layout.addLayout(stat_row)

    # ── Imbalance warning ────────────────────────────────────────────────────────
    if label_counts and len(label_counts) > 1:
        max_c      = max(label_counts)
        imbalanced = [label_names[i] for i, c in enumerate(label_counts)
                      if max_c > 0 and c < max_c * 0.1]
        if imbalanced:
            warn_label = QtWidgets.QLabel(
                "⚠  Imbalanced classes (< 10 % of max):  " + ",  ".join(imbalanced)
                + "\n   Low-sample classes are shown in red on the chart."
                   "  Consider adding more annotations."
            )
            warn_label.setStyleSheet(
                "color: #ffcc00; background: #3a2a00; border-radius: 8px;"
                " padding: 10px 14px; font-size: 13px;"
            )
            warn_label.setWordWrap(True)
            warn_label.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Preferred,
            )
            main_layout.addWidget(warn_label)

    # ── Label order card ─────────────────────────────────────────────────────────
    label_card = QtWidgets.QFrame()
    label_card.setFrameShape(QtWidgets.QFrame.StyledPanel)
    label_card.setStyleSheet("""
        QFrame {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #28283a, stop:1 #1e1e2a);
            border-radius: 20px;
            border: 1px solid #444458;
        }
    """)
    label_card.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

    label_card_layout = QtWidgets.QVBoxLayout(label_card)
    label_card_layout.setContentsMargins(24, 20, 24, 20)
    label_card_layout.setSpacing(12)

    title_lbl = QtWidgets.QLabel("Labels Order in YOLOv5 Dataset")
    title_lbl.setAlignment(QtCore.Qt.AlignCenter)
    title_lbl.setStyleSheet("""
        font-size: 16px; color: #ffffff; font-weight: 400;
        letter-spacing: 2px; font-family: Arial;
    """)
    title_lbl.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    label_card_layout.addWidget(title_lbl)

    label_order = QtWidgets.QLabel(" → ".join(label_names))
    label_order.setStyleSheet("""
        font-size: 18px; font-weight: bold; color: #C1FFB5;
        letter-spacing: 1px; font-family: Arial Rounded MT Bold, Arial, sans-serif;
    """)
    label_order.setWordWrap(True)
    label_order.setAlignment(QtCore.Qt.AlignCenter)
    label_order.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    label_card_layout.addWidget(label_order)

    main_layout.addWidget(label_card)

    # ── Charts ───────────────────────────────────────────────────────────────────
    # Size the figure relative to the dialog so it never overflows
    fig_w = max(6,  (dlg_w - 60) / 100)   # convert px → inches (approx 100 dpi)
    fig_h = max(3.5, dlg_h * 0.38 / 100)

    _bg  = '#1e1e24'
    _bg2 = '#28283a'
    plt.style.use('dark_background')
    fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h), facecolor=_bg)

    axs[0].pie(
        [num_train, num_val], labels=["Train", "Val"], autopct='%1.1f%%',
        startangle=90, colors=["#5558a0", "#888adf"], textprops={'color': '#ddd'},
    )
    axs[0].set_title("Train / Val Split", color='#ddd', fontsize=12)
    axs[0].set_facecolor(_bg)

    max_count  = max(label_counts) if label_counts else 1
    bar_colors = ["#e05555" if c < max_count * 0.1 else "#888adf" for c in label_counts]
    axs[1].barh(label_names, label_counts, color=bar_colors)
    axs[1].set_title("Annotations per Label", color='#ddd', fontsize=12)
    axs[1].set_xlabel("Count", color='#aaa')
    axs[1].tick_params(axis='x', colors='#aaa')
    axs[1].tick_params(axis='y', colors='#ddd')
    axs[1].set_facecolor(_bg2)
    for spine in ['bottom', 'left']:
        axs[1].spines[spine].set_color('#444458')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].invert_yaxis()

    plt.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    main_layout.addWidget(canvas)

    dialog.exec_()
    plt.close(fig)
