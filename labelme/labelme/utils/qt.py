import os.path as osp
from math import sqrt

import numpy as np
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

here = osp.dirname(osp.abspath(__file__))


def newIcon(icon):
    icons_dir = osp.join(here, "../icons")
    return QtGui.QIcon(osp.join(":/", icons_dir, "%s.png" % icon))


def newButton(text, icon=None, slot=None):
    b = QtWidgets.QPushButton(text)
    if icon is not None:
        b.setIcon(newIcon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def newAction(
    parent,
    text,
    slot=None,
    shortcut=None,
    icon=None,
    tip=None,
    checkable=False,
    enabled=True,
    checked=False,
):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QtWidgets.QAction(text, parent)
    if icon is not None:
        a.setIconText(text.replace(" ", "\n"))
        a.setIcon(newIcon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    a.setChecked(checked)
    return a


def addActions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QtWidgets.QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def labelValidator():
    return QtGui.QRegExpValidator(QtCore.QRegExp(r"^[^ \t].+"), None)


class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def distancetoline(point, line):
    p1, p2 = line
    p1 = np.array([p1.x(), p1.y()])
    p2 = np.array([p2.x(), p2.y()])
    p3 = np.array([point.x(), point.y()])
    if np.dot((p3 - p1), (p2 - p1)) < 0:
        return np.linalg.norm(p3 - p1)
    if np.dot((p3 - p2), (p1 - p2)) < 0:
        return np.linalg.norm(p3 - p2)
    if np.linalg.norm(p2 - p1) == 0:
        return np.linalg.norm(p3 - p1)
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fmtShortcut(text):
    mod, key = text.split("+", 1)
    return "<b>%s</b>+<b>%s</b>" % (mod, key)


def _screen_for(widget):
    """Return the QScreen that contains *widget* (or its nearest visible ancestor).

    Falls back to the primary screen if nothing better can be found.
    This is the correct way to handle multi-monitor setups: always use the
    screen where the parent window actually lives, not screen 0.
    """
    app = QtWidgets.QApplication.instance()
    w = widget
    while w is not None:
        try:
            visible = w.isVisible()
        except Exception:
            visible = False
        if visible:
            screen = app.screenAt(w.frameGeometry().center())
            if screen is not None:
                return screen
        try:
            w = w.parent()
        except Exception:
            break
    return app.primaryScreen()


def fit_dialog(dialog, desired_w, desired_h, w_pct=0.92, h_pct=0.90):
    """Resize *dialog* to fit the screen that its parent window lives on.

    The dialog will be at most *desired_w* × *desired_h*, but never larger
    than *w_pct* × *h_pct* of that screen's available geometry.
    After resizing the dialog is centered on that screen.
    """
    screen = _screen_for(dialog).availableGeometry()
    w = min(desired_w, int(screen.width()  * w_pct))
    h = min(desired_h, int(screen.height() * h_pct))
    dialog.resize(w, h)
    dialog.move(
        screen.x() + (screen.width()  - w) // 2,
        screen.y() + (screen.height() - h) // 2,
    )
