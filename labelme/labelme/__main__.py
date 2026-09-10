import sys
import os

# PyInstaller with console=False sets sys.stdout/sys.stderr to None.
# Any code that calls print() or writes to stdout/stderr (including ultralytics
# training output and spawned worker processes) will crash with:
#   "NoneType object has no attribute 'write'"
# Redirect to devnull so those writes silently succeed.
# Must happen before ANY other import that might print on load.
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

# CRITICAL: Set CUDA environment variable BEFORE any torch imports
# This prevents deterministic CUDA errors during training/validation
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import codecs
import contextlib
import io
import os.path as osp
import sys
import traceback

# CRITICAL: Multiprocessing freeze support for PyInstaller
# This prevents "--multiprocessing-fork pipe_handle" errors when using PyTorch DataLoader
import multiprocessing
multiprocessing.freeze_support()

import yaml
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

from labelme import __appname__
from labelme import __version__
from labelme.app import MainWindow
from labelme.config import get_config
from labelme.utils import newIcon


class _LoggerIO(io.StringIO):
    def write(self, message: str) -> int:
        if stripped_message := message.strip():
            logger.debug(stripped_message)
        return len(message)

    def flush(self) -> None:
        pass

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return False


def _setup_loguru(logger_level: str) -> None:
    try:
        logger.remove(handler_id=0)
    except ValueError:
        pass

    if sys.stderr:
        logger.add(sys.stderr, level=logger_level)

    cache_dir: str
    if os.name == "nt":
        cache_dir = os.path.join(os.environ["LOCALAPPDATA"], "labelme")
    else:
        cache_dir = os.path.expanduser("~/.cache/labelme")

    os.makedirs(cache_dir, exist_ok=True)

    log_file = os.path.join(cache_dir, "labelme.log")
    logger.add(
        log_file,
        colorize=True,
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )


def _handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        sys.exit(0)

    traceback_str: str = "".join(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    )
    logger.critical(traceback_str)

    traceback_html: str = traceback_str.replace("\n", "<br/>").replace(" ", "&nbsp;")
    QtWidgets.QMessageBox.critical(
        None,
        "Error",
        f"An unexpected error occurred. The application will close.<br/><br/>Please report issues following the <a href='https://labelme.io/docs/troubleshoot'>Troubleshoot</a>.<br/><br/>{traceback_html}",  # noqa: E501
    )

    if app := QtWidgets.QApplication.instance():
        app.quit()
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", "-V", action="store_true", help="show version")
    parser.add_argument("--reset-config", action="store_true", help="reset qt config")
    parser.add_argument(
        "--logger-level",
        default="debug",
        choices=["debug", "info", "warning", "fatal", "error"],
        help="logger level",
    )
    parser.add_argument("filename", nargs="?", help="image or label filename")
    parser.add_argument(
        "--output",
        "-O",
        "-o",
        help="output file or directory (if it ends with .json it is "
        "recognized as file, else as directory)",
    )
    default_config_file = os.path.join(os.path.expanduser("~"), ".labelmerc")
    parser.add_argument(
        "--config",
        dest="config",
        help="config file or yaml-format string (default: {})".format(
            default_config_file
        ),
        default=default_config_file,
    )
    # config for the gui
    parser.add_argument(
        "--nodata",
        dest="store_data",
        action="store_false",
        help="stop storing image data to JSON file",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--autosave",
        dest="auto_save",
        action="store_true",
        help="auto save",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--nosortlabels",
        dest="sort_labels",
        action="store_false",
        help="stop sorting labels",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--flags",
        help="comma separated list of flags OR file containing flags",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--labelflags",
        dest="label_flags",
        help=r"yaml string of label specific flags OR file containing json "
        r"string of label specific flags (ex. {person-\d+: [male, tall], "
        r"dog-\d+: [black, brown, white], .*: [occluded]})",  # NOQA
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--labels",
        help="comma separated list of labels OR file containing labels",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--validatelabel",
        dest="validate_label",
        choices=["exact"],
        help="label validation types",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--keep-prev",
        action="store_true",
        help="keep annotation of previous frame",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="epsilon to find nearest vertex on canvas",
        default=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.version:
        print("{0} {1}".format(__appname__, __version__))
        sys.exit(0)

    _setup_loguru(logger_level=args.logger_level.upper())

    sys.excepthook = _handle_exception

    if hasattr(args, "flags"):
        if os.path.isfile(args.flags):
            with codecs.open(args.flags, "r", encoding="utf-8") as f:
                args.flags = [line.strip() for line in f if line.strip()]
        else:
            args.flags = [line for line in args.flags.split(",") if line]

    if hasattr(args, "labels"):
        if os.path.isfile(args.labels):
            with codecs.open(args.labels, "r", encoding="utf-8") as f:
                args.labels = [line.strip() for line in f if line.strip()]
        else:
            args.labels = [line for line in args.labels.split(",") if line]

    if hasattr(args, "label_flags"):
        if os.path.isfile(args.label_flags):
            with codecs.open(args.label_flags, "r", encoding="utf-8") as f:
                args.label_flags = yaml.safe_load(f)
        else:
            args.label_flags = yaml.safe_load(args.label_flags)

    config_from_args = args.__dict__
    config_from_args.pop("version")
    reset_config = config_from_args.pop("reset_config")
    filename = config_from_args.pop("filename")
    output = config_from_args.pop("output")
    config_file_or_yaml = config_from_args.pop("config")
    config = get_config(config_file_or_yaml, config_from_args)

    if not config["labels"] and config["validate_label"]:
        logger.error(
            "--labels must be specified with --validatelabel or "
            "validate_label: true in the config file "
            "(ex. ~/.labelmerc)."
        )
        sys.exit(1)

    output_file = None
    output_dir = None
    if output is not None:
        if output.endswith(".json"):
            output_file = output
        else:
            output_dir = output

    # Enable HiDPI scaling BEFORE QApplication is created.
    # Without these, Qt5 uses physical pixels for geometry on Windows, while
    # Windows itself enforces DPI-scaled (logical) pixel limits — causing
    # "Unable to set geometry" warnings on scaled monitors (e.g. 125% DPI).
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps,    True)

    translator = QtCore.QTranslator()
    translator.load(
        QtCore.QLocale.system().name(),
        osp.dirname(osp.abspath(__file__)) + "/translate",
    )
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("icon"))
    app.installTranslator(translator)
    
    # ── Theme ────────────────────────────────────────────────────────────────
    app.setStyle("Fusion")

    # Palette — base colours that Qt uses for native drawing
    pal = QtGui.QPalette()
    _bg      = QtGui.QColor(0x1e, 0x1e, 0x24)   # #1e1e24  main background
    _bg2     = QtGui.QColor(0x28, 0x28, 0x30)   # #282830  input / list bg
    _bg3     = QtGui.QColor(0x30, 0x30, 0x3a)   # #30303a  buttons
    _border  = QtGui.QColor(0x50, 0x50, 0x60)   # #505060  borders
    _accent  = QtGui.QColor(0x88, 0x8a, 0xdf)   # #888adf  purple accent
    _accent2 = QtGui.QColor(0x55, 0x58, 0xa0)   # #5558a0  pressed / selected
    _text    = QtGui.QColor(0xdd, 0xdd, 0xdd)   # #dddddd
    _dim     = QtGui.QColor(0x77, 0x77, 0x88)   # #777788  disabled
    pal.setColor(QtGui.QPalette.Window,           _bg)
    pal.setColor(QtGui.QPalette.WindowText,       _text)
    pal.setColor(QtGui.QPalette.Base,             _bg2)
    pal.setColor(QtGui.QPalette.AlternateBase,    _bg3)
    pal.setColor(QtGui.QPalette.ToolTipBase,      _bg)
    pal.setColor(QtGui.QPalette.ToolTipText,      _text)
    pal.setColor(QtGui.QPalette.Text,             _text)
    pal.setColor(QtGui.QPalette.Button,           _bg3)
    pal.setColor(QtGui.QPalette.ButtonText,       _text)
    pal.setColor(QtGui.QPalette.BrightText,       QtGui.QColor("#ff6b6b"))
    pal.setColor(QtGui.QPalette.Link,             _accent)
    pal.setColor(QtGui.QPalette.Highlight,        _accent2)
    pal.setColor(QtGui.QPalette.HighlightedText,  QtGui.QColor("#ffffff"))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText,     _dim)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text,           _dim)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText,     _dim)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Highlight,      _bg3)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.HighlightedText,_dim)
    app.setPalette(pal)

    app.setStyleSheet("""
/* ── Global ────────────────────────────────────────────────────────────── */
* { font-family: "Segoe UI", sans-serif; font-size: 12px; color: #dddddd; }

QWidget            { background: #1e1e24; }
QMainWindow        { background: #1e1e24; }

/* ── Tooltip ────────────────────────────────────────────────────────────── */
QToolTip {
    background: #28283a; color: #eee;
    border: 1px solid #888adf; border-radius: 4px; padding: 4px 8px;
}

/* ── Menu bar ───────────────────────────────────────────────────────────── */
QMenuBar { background: #18181e; color: #ccc; }
QMenuBar::item { padding: 4px 12px; background: transparent; }
QMenuBar::item:selected { background: #5558a0; color: #fff; border-radius: 3px; }

QMenu { background: #222230; border: 1px solid #444458; padding: 4px; }
QMenu::item { padding: 5px 28px 5px 16px; border-radius: 3px; }
QMenu::item:selected { background: #5558a0; color: #fff; }
QMenu::separator { height: 1px; background: #3a3a4a; margin: 3px 8px; }

/* ── Toolbar ────────────────────────────────────────────────────────────── */
QToolBar { background: #1a1a22; border: none; spacing: 3px; padding: 2px; }
QToolButton {
    background: transparent; border: 1px solid transparent;
    border-radius: 5px; padding: 3px;
}
QToolButton:hover   { background: #30303a; border-color: #505060; }
QToolButton:pressed { background: #5558a0; }
QToolButton:checked { background: #3a3a50; border-color: #888adf; }

/* ── Status bar ─────────────────────────────────────────────────────────── */
QStatusBar { background: #18181e; color: #999; border-top: 1px solid #333340; }

/* ── Dock widgets ───────────────────────────────────────────────────────── */
QDockWidget { background: #1e1e24; color: #ccc; }
QDockWidget::title {
    background: #28283a; padding: 5px 8px;
    border-bottom: 1px solid #444458; text-align: left;
}

/* ── Dialogs & message boxes ────────────────────────────────────────────── */
QDialog, QMessageBox, QProgressDialog, QFileDialog {
    background: #1e1e24;
    border: 1px solid #444458;
    border-radius: 8px;
}
QMessageBox QLabel { color: #ddd; font-size: 12px; }
QMessageBox QPushButton { min-width: 80px; }

/* ── Buttons ────────────────────────────────────────────────────────────── */
QPushButton {
    background: #30303a; color: #ddd;
    border: 1px solid #505060;
    border-radius: 5px; padding: 5px 16px;
    min-height: 22px;
}
QPushButton:hover   { background: #3c3c4a; border-color: #888adf; color: #fff; }
QPushButton:pressed { background: #5558a0; border-color: #888adf; color: #fff; }
QPushButton:checked { background: #3a3a54; border-color: #888adf; color: #fff; }
QPushButton:disabled{ background: #262630; color: #555566; border-color: #383848; }
QPushButton[default="true"] {
    border-color: #888adf; background: #5558a0; color: #fff;
}
QPushButton[default="true"]:hover { background: #6668b8; }

/* ── Inputs ─────────────────────────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {
    background: #222230; color: #eee;
    border: 1px solid #444458; border-radius: 5px; padding: 4px 8px;
    selection-background-color: #5558a0;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #888adf;
}
QSpinBox, QDoubleSpinBox {
    background: #222230; color: #eee;
    border: 1px solid #444458; border-radius: 5px; padding: 3px 6px;
}
QSpinBox:focus, QDoubleSpinBox:focus { border-color: #888adf; }

QComboBox {
    background: #222230; color: #eee;
    border: 1px solid #444458; border-radius: 5px; padding: 4px 8px;
}
QComboBox:focus { border-color: #888adf; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #222230; border: 1px solid #444458;
    selection-background-color: #5558a0;
}

/* ── Sliders ────────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 4px; background: #383848; border-radius: 2px;
}
QSlider::sub-page:horizontal { background: #5558a0; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #888adf; border: 2px solid #1e1e24;
    width: 16px; height: 16px; margin: -6px 0; border-radius: 8px;
}
QSlider::handle:horizontal:hover { background: #aaaaff; }

/* ── Progress bar ───────────────────────────────────────────────────────── */
QProgressBar {
    background: #282838; border: 1px solid #444458;
    border-radius: 6px; text-align: center; color: #ccc;
    min-height: 14px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #5558a0, stop:1 #888adf);
    border-radius: 5px;
}

/* ── Checkboxes & radio buttons ─────────────────────────────────────────── */
QCheckBox, QRadioButton { color: #ccc; spacing: 6px; }
QCheckBox::indicator, QRadioButton::indicator {
    width: 15px; height: 15px;
    background: #222230; border: 1px solid #505060; border-radius: 3px;
}
QRadioButton::indicator { border-radius: 8px; }
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background: #5558a0; border-color: #888adf;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: #888adf; }

/* ── Group box ──────────────────────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #3a3a4a; border-radius: 6px;
    margin-top: 14px; padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 10px; padding: 0 6px; color: #888adf;
}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
QTabWidget::pane { border: 1px solid #3a3a4a; background: #1e1e24; }
QTabBar::tab {
    background: #252530; color: #999;
    border: 1px solid #3a3a4a; border-bottom: none;
    padding: 5px 14px; border-radius: 5px 5px 0 0;
    margin-right: 2px;
}
QTabBar::tab:selected { background: #1e1e24; color: #fff; border-color: #5558a0; }
QTabBar::tab:hover    { background: #2e2e3a; color: #ddd; }

/* ── Scrollbars ─────────────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #1a1a22; width: 10px; margin: 0;
}
QScrollBar::handle:vertical {
    background: #404050; border-radius: 5px; min-height: 24px; margin: 1px;
}
QScrollBar::handle:vertical:hover { background: #5558a0; }
QScrollBar:horizontal {
    background: #1a1a22; height: 10px; margin: 0;
}
QScrollBar::handle:horizontal {
    background: #404050; border-radius: 5px; min-width: 24px; margin: 1px;
}
QScrollBar::handle:horizontal:hover { background: #5558a0; }
QScrollBar::add-line, QScrollBar::sub-line { border: none; background: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

/* ── Lists & tables ─────────────────────────────────────────────────────── */
QListWidget, QListView, QTreeWidget, QTreeView {
    background: #222230; border: 1px solid #3a3a4a;
    border-radius: 5px; outline: none;
}
QListWidget::item, QListView::item { padding: 3px 6px; border-radius: 3px; }
QListWidget::item:selected, QListView::item:selected {
    background: #5558a0; color: #fff;
}
QListWidget::item:hover, QListView::item:hover { background: #2e2e3e; }

QTableWidget, QTableView {
    background: #222230; gridline-color: #333344;
    border: 1px solid #3a3a4a; border-radius: 5px;
}
QHeaderView::section {
    background: #2a2a38; color: #bbb;
    border: none; border-right: 1px solid #3a3a4a;
    padding: 5px 8px; font-weight: bold;
}

/* ── Splitters ──────────────────────────────────────────────────────────── */
QSplitter::handle { background: #2a2a38; }
QSplitter::handle:horizontal { width: 2px; }
QSplitter::handle:vertical   { height: 2px; }

/* ── Label (plain) ──────────────────────────────────────────────────────── */
QLabel { background: transparent; }
    """)

    # ── Frameless dialogs ─────────────────────────────────────────────────
    # Three-class system that covers ALL QDialogs (QMessageBox, QProgressDialog,
    # startup dialogs, training dialogs, etc.) except the native QFileDialog.
    #
    # _HeaderBar       — draggable overlay strip: title label + big ✕ button
    # _FramelessDecorator — per-dialog event filter: sets flags, mask, margin
    # _GlobalFramelessFilter — app-level filter: installs decorator on first Show

    _HEADER_H   = 46   # px — height of the injected header strip
    _BTN_SIZE   = 38   # px — close button size
    _CORNER_R   = 6    # px — window corner radius

    class _XButton(QtWidgets.QAbstractButton):
        """Close button that paints its own X — never clipped by font rendering."""

        def __init__(self, dialog, parent):
            super().__init__(parent)
            self._dialog = dialog
            self.setFixedSize(_BTN_SIZE, _BTN_SIZE)
            self.setCursor(QtCore.Qt.PointingHandCursor)
            self.clicked.connect(dialog.reject)
            self.setAttribute(QtCore.Qt.WA_Hover, True)

        def paintEvent(self, _ev):
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            hover = self.underMouse()
            bg  = QtGui.QColor("#c0392b" if hover else "#3a3050")
            fg  = QtGui.QColor("#ffffff" if hover else "#ccccee")
            bdr = QtGui.QColor("#ff4444" if hover else "#555577")
            p.setBrush(bg)
            p.setPen(QtGui.QPen(bdr, 1))
            p.drawRoundedRect(1, 1, _BTN_SIZE - 2, _BTN_SIZE - 2, 5, 5)
            pad = _BTN_SIZE // 4
            p.setPen(QtGui.QPen(fg, 2.5,
                                QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            p.drawLine(pad, pad, _BTN_SIZE - pad, _BTN_SIZE - pad)
            p.drawLine(_BTN_SIZE - pad, pad, pad, _BTN_SIZE - pad)
            p.end()

        def sizeHint(self):
            return QtCore.QSize(_BTN_SIZE, _BTN_SIZE)

    class _HeaderBar(QtWidgets.QWidget):
        """Header strip: title + X button.  Fixed height, expands horizontally."""

        def __init__(self, dialog):
            super().__init__(dialog)
            self._dialog   = dialog
            self._drag_pos = None

            self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
            self.setStyleSheet("background: #23233a;")
            self.setFixedHeight(_HEADER_H)
            self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                               QtWidgets.QSizePolicy.Fixed)

            lay = QtWidgets.QHBoxLayout(self)
            lay.setContentsMargins(14, 4, 8, 4)
            lay.setSpacing(0)

            self._title = QtWidgets.QLabel(dialog.windowTitle())
            self._title.setStyleSheet(
                "font-size:13px; font-weight:600;"
                "color:#aaaadf; background:transparent;"
            )
            lay.addWidget(self._title, 1)
            lay.addWidget(_XButton(dialog, self))

        def mousePressEvent(self, ev):
            if ev.button() == QtCore.Qt.LeftButton:
                self._drag_pos = (
                    ev.globalPos() - self._dialog.frameGeometry().topLeft()
                )
            super().mousePressEvent(ev)

        def mouseMoveEvent(self, ev):
            if self._drag_pos and ev.buttons() & QtCore.Qt.LeftButton:
                self._dialog.move(ev.globalPos() - self._drag_pos)
            super().mouseMoveEvent(ev)

        def mouseReleaseEvent(self, ev):
            self._drag_pos = None
            super().mouseReleaseEvent(ev)

        def sync_title(self):
            self._title.setText(self._dialog.windowTitle())

    class _FramelessDecorator(QtCore.QObject):
        """Per-dialog event filter: applies frameless flags, rounded mask, header."""

        def __init__(self, dialog):
            super().__init__(dialog)
            self._dialog      = dialog
            self._header      = None
            self._applied     = False
            self._use_overlay = False   # True = absolute pos, False = in layout

        def _apply(self):
            if self._applied:
                return
            self._applied = True
            dlg     = self._dialog
            _parent = dlg.parent()

            # Screen limits — use the screen where the parent window lives,
            # not always screen 0 (important for multi-monitor setups).
            def _parent_screen(widget):
                app = QtWidgets.QApplication.instance()
                w = widget
                while w is not None:
                    try:
                        if w.isVisible():
                            s = app.screenAt(w.frameGeometry().center())
                            if s is not None:
                                return s
                    except Exception:
                        pass
                    try:
                        w = w.parent()
                    except Exception:
                        break
                return app.primaryScreen()

            screen = _parent_screen(_parent).availableGeometry()
            max_w  = int(screen.width()  * 0.92)
            max_h  = int(screen.height() * 0.90)

            # 1. Remove OS title bar
            dlg.setWindowFlags(
                QtCore.Qt.Dialog |
                QtCore.Qt.FramelessWindowHint |
                QtCore.Qt.WindowSystemMenuHint
            )

            # 2. Show so dialog has its natural laid-out size
            dlg.show()
            dlg.raise_()

            # 3. Build header
            self._header = _HeaderBar(dlg)
            layout = dlg.layout()

            if isinstance(layout, QtWidgets.QBoxLayout):
                # ── Best case: insert header as first layout item ──────
                self._use_overlay = False
                layout.insertWidget(0, self._header)
                layout.setSpacing(max(layout.spacing(), 0))
                # Clear any fixed-size constraints so the window can grow.
                dlg.setMinimumHeight(0)
                dlg.setMaximumHeight(16777215)
                new_w = min(dlg.width(),              max_w)
                new_h = min(dlg.height() + _HEADER_H, max_h)
                dlg.resize(new_w, new_h)
            else:
                # ── Fallback: absolute overlay (e.g. QMessageBox) ──────
                self._use_overlay = True
                orig_top = layout.contentsMargins().top() if layout else 0
                new_top  = max(orig_top, _HEADER_H + 8)
                if layout:
                    m = layout.contentsMargins()
                    layout.setContentsMargins(
                        m.left(), new_top, m.right(), m.bottom())
                    layout.activate()

                # Enable word-wrap on all labels so long messages wrap instead
                # of pushing the dialog off the edge of the screen.
                label_max_w = min(max_w - 120, 620)
                for lbl in dlg.findChildren(QtWidgets.QLabel):
                    if lbl.text():
                        lbl.setWordWrap(True)
                        lbl.setMaximumWidth(label_max_w)

                # Clear any fixed-size constraints (QMessageBox sets setFixedSize)
                dlg.setMinimumSize(0, 0)
                dlg.setMaximumSize(16777215, 16777215)
                # Let Qt recompute size after margin + word-wrap changes, then cap
                dlg.adjustSize()
                new_w = max(min(dlg.width(),  max_w), 320)
                new_h = max(min(dlg.height(), max_h), _HEADER_H + 100)
                dlg.resize(new_w, new_h)
                self._header.move(0, 0)
                self._header.resize(new_w, _HEADER_H)
                self._header.raise_()
                self._header.show()
                self._header.raise_()

            # 4. Rounded corners
            self._apply_mask()

            # 5. Center on parent (or the correct screen), always fully visible
            if _parent and _parent.isVisible():
                c = _parent.frameGeometry().center()
            else:
                c = screen.center()   # already the right screen from step above
            w, h = dlg.width(), dlg.height()
            x = max(screen.x(), min(c.x() - w // 2, screen.right()  - w))
            y = max(screen.y(), min(c.y() - h // 2, screen.bottom() - h))
            dlg.move(x, y)

        def _apply_mask(self):
            dlg  = self._dialog
            path = QtGui.QPainterPath()
            path.addRoundedRect(QtCore.QRectF(dlg.rect()), _CORNER_R, _CORNER_R)
            dlg.setMask(QtGui.QRegion(path.toFillPolygon().toPolygon()))

        def _update_header(self):
            if not self._header:
                return
            if self._use_overlay:
                self._header.resize(self._dialog.width(), _HEADER_H)
            self._header.sync_title()

        def eventFilter(self, obj, ev):
            if obj is self._dialog:
                t = ev.type()
                if t == QtCore.QEvent.Show:
                    if not self._applied:
                        self._apply()
                elif t == QtCore.QEvent.Resize:
                    if self._applied:
                        self._update_header()
                        self._apply_mask()
                elif t == QtCore.QEvent.WindowTitleChange:
                    if self._header:
                        self._header.sync_title()
            return False

    class _GlobalFramelessFilter(QtCore.QObject):
        """App-level event filter: installs _FramelessDecorator on every QDialog."""

        def eventFilter(self, obj, ev):
            if (
                ev.type() == QtCore.QEvent.Show
                and isinstance(obj, QtWidgets.QDialog)
                and not isinstance(obj, QtWidgets.QFileDialog)
                and not getattr(obj, '_frameless_done', False)
            ):
                obj._frameless_done = True
                dec = _FramelessDecorator(obj)
                obj.installEventFilter(dec)
                dec._apply()
            return False

    _gff = _GlobalFramelessFilter(app)
    app.installEventFilter(_gff)

    win = MainWindow(
        config=config,
        filename=filename,
        output_file=output_file,
        output_dir=output_dir,
    )

    if reset_config:
        logger.info("Resetting Qt config: %s" % win.settings.fileName())
        win.settings.clear()
        sys.exit(0)

    with contextlib.redirect_stderr(new_target=_LoggerIO()):
        win.show()
        win.raise_()
        sys.exit(app.exec_())


# this main block is required to generate executable by pyinstaller
if __name__ == "__main__":
    # Freeze support is critical for PyInstaller with multiprocessing
    multiprocessing.freeze_support()
    main()
