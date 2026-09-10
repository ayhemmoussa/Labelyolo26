from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

class TrainValSplitDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Train/Val Split")
        self.setFixedWidth(350)
        layout = QtWidgets.QVBoxLayout(self)

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(99)
        self.slider.setValue(80)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        layout.addWidget(self.slider)

        self.label = QtWidgets.QLabel("Train: 80%   Val: 20%")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.slider.valueChanged.connect(self.update_label)

        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)

    def update_label(self, value):
        self.label.setText(f"Train: {value}%   Val: {100-value}%")

    def get_split(self):
        return self.slider.value(), 100 - self.slider.value()