from typing import cast

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class LabelListWidgetItem(QtGui.QStandardItem):
    def __init__(self, text=None, shape=None):
        super(LabelListWidgetItem, self).__init__()
        self.setText(text or "")
        self.setShape(shape)

        self.setCheckable(True)
        self.setCheckState(Qt.Checked)  # type: ignore[attr-defined]
        self.setEditable(False)
        self.setTextAlignment(Qt.AlignVCenter)  # type: ignore[attr-defined]
        if text:
            self.setToolTip(text)

    def clone(self):
        return LabelListWidgetItem(self.text(), self.shape())

    def setShape(self, shape):
        self.setData(shape, Qt.UserRole)  # type: ignore[attr-defined]

    def shape(self):
        return self.data(Qt.UserRole)  # type: ignore[attr-defined]

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return '{}("{}")'.format(self.__class__.__name__, self.text())


class StandardItemModel(QtGui.QStandardItemModel):
    itemDropped = QtCore.pyqtSignal()

    def removeRows(self, *args, **kwargs):
        ret = super().removeRows(*args, **kwargs)
        self.itemDropped.emit()
        return ret


class LabelListWidget(QtWidgets.QListView):
    itemDoubleClicked = QtCore.pyqtSignal(LabelListWidgetItem)
    itemSelectionChanged = QtCore.pyqtSignal(list, list)

    def __init__(self):
        super(LabelListWidget, self).__init__()
        self._selectedItems = []

        self.setWindowFlags(Qt.Window)  # type: ignore[attr-defined]

        self._model: StandardItemModel = StandardItemModel()
        self._model.setItemPrototype(LabelListWidgetItem())  # type: ignore[union-attr]
        self.setModel(self._model)

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)  # type: ignore[attr-defined]
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        self.setWordWrap(False)
        self.setMinimumWidth(200)
        self.setIconSize(QtCore.QSize(14, 14))

        self.doubleClicked.connect(self.itemDoubleClickedEvent)
        self.selectionModel().selectionChanged.connect(self.itemSelectionChangedEvent)  # type: ignore[union-attr]

    def __len__(self):
        return self._model.rowCount()  # type: ignore[union-attr]

    def __getitem__(self, i):
        return self._model.item(i)  # type: ignore[union-attr]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def itemDropped(self):
        return self._model.itemDropped  # type: ignore[union-attr]

    @property
    def itemChanged(self):
        return self._model.itemChanged  # type: ignore[union-attr]

    def itemSelectionChangedEvent(self, selected, deselected):
        selected = [self._model.itemFromIndex(i) for i in selected.indexes()]  # type: ignore[union-attr]
        deselected = [self._model.itemFromIndex(i) for i in deselected.indexes()]  # type: ignore[union-attr]
        self.itemSelectionChanged.emit(selected, deselected)

    def itemDoubleClickedEvent(self, index):
        self.itemDoubleClicked.emit(self._model.itemFromIndex(index))  # type: ignore[union-attr]

    def selectedItems(self):
        return [self._model.itemFromIndex(i) for i in self.selectedIndexes()]  # type: ignore[union-attr]

    def scrollToItem(self, item):
        self.scrollTo(self._model.indexFromItem(item))  # type: ignore[union-attr]

    def addItem(self, item):
        if not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem")
        self._model.setItem(self._model.rowCount(), 0, item)  # type: ignore[union-attr]
        item.setSizeHint(QtCore.QSize(0, 28))

    def setItemLabel(self, item, label_text, color=None):
        """Set plain text colored in the label's color."""
        item.setText(label_text)
        item.setToolTip(label_text)
        item.setIcon(QtGui.QIcon())  # clear any previous icon
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        if color is not None:
            r, g, b = color[0], color[1], color[2]
            item.setForeground(QtGui.QColor(r, g, b))

    def removeItem(self, item):
        index = self._model.indexFromItem(item)  # type: ignore[union-attr]
        self._model.removeRows(index.row(), 1)  # type: ignore[union-attr]

    def selectItem(self, item):
        index = self._model.indexFromItem(item)  # type: ignore[union-attr]
        self.selectionModel().select(index, QtCore.QItemSelectionModel.Select)  # type: ignore[attr-defined,union-attr]

    def findItemByShape(self, shape):
        for row in range(self._model.rowCount()):  # type: ignore[union-attr]
            item = self._model.item(row, 0)  # type: ignore[union-attr]
            item = cast(LabelListWidgetItem, item)
            if item.shape() == shape:
                return item
        raise ValueError("cannot find shape: {}".format(shape))

    def clear(self):
        self._model.clear()  # type: ignore[union-attr]
