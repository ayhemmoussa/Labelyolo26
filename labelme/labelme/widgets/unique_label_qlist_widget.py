# -*- encoding: utf-8 -*-

from cProfile import label
import html

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from .escapable_qlist_widget import EscapableQListWidget


class UniqueLabelQListWidget(EscapableQListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        self.setWordWrap(False)
        self.setMinimumWidth(200)

    def mousePressEvent(self, event):
        super(UniqueLabelQListWidget, self).mousePressEvent(event)
        if not self.indexAt(event.pos()).isValid():
            self.clearSelection()

    def findItemByLabel(self, label):
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.UserRole) == label:  # type: ignore[attr-defined,union-attr]
                return item

    def createItemFromLabel(self, label):
        if self.findItemByLabel(label):
            raise ValueError("Item for label '{}' already exists".format(label))

        item = QtWidgets.QListWidgetItem(label)
        item.setData(Qt.UserRole, label)  # type: ignore[attr-defined]
        return item

    def setItemLabel(self, item, label, color=None, count=None):
        # Remove any existing widget to avoid stacking
        old_widget = self.itemWidget(item)
        if old_widget is not None:
            self.removeItemWidget(item)
            old_widget.deleteLater()

        qlabel = QtWidgets.QLabel()
        qlabel.setWordWrap(False)
        count_str = (
            ' <span style="color: #888; font-size: 10px;">({})</span>'.format(count)
            if count is not None
            else ''
        )
        if color is None:
            qlabel.setText('{}{}'.format(html.escape(label), count_str))
        else:
            r, g, b = color[0], color[1], color[2]
            qlabel.setText(
                '<span style="color: #{:02x}{:02x}{:02x}; font-weight: 600;">{}</span>{}'.format(
                    r, g, b, html.escape(label), count_str
                )
            )
        qlabel.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)  # type: ignore[attr-defined]
        qlabel.setContentsMargins(4, 0, 4, 0)
        qlabel.setToolTip(label)
        natural_w = qlabel.sizeHint().width() + 8
        qlabel.setMinimumWidth(natural_w)
        item.setText("")
        item.setSizeHint(QtCore.QSize(natural_w, 24))

        self.setItemWidget(item, qlabel)
