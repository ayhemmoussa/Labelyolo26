import PIL.Image
import PIL.ImageEnhance
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage


class BrightnessContrastDialog(QtWidgets.QDialog):
    _base_value = 50

    def __init__(self, img, callback, parent=None):
        super(BrightnessContrastDialog, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Brightness/Contrast")
        self.setMinimumWidth(520)
        self.setMinimumHeight(120)

        assert isinstance(img, PIL.Image.Image)
        self.img = img
        self.callback = callback
        # Keep a QImage of the unmodified original so Cancel can restore it
        self._original_qimage = QImage(
            img.tobytes(), img.width, img.height, img.width * 3, QImage.Format_RGB888
        )

        sliders = {}
        layouts = {}
        for title in ["Brightness:", "Contrast:"]:
            layout = QtWidgets.QHBoxLayout()
            title_label = QtWidgets.QLabel(self.tr(title))
            title_label.setFixedWidth(75)
            layout.addWidget(title_label)
            #
            slider = QtWidgets.QSlider(Qt.Horizontal)  # type: ignore[attr-defined]
            slider.setRange(0, 3 * self._base_value)
            slider.setValue(self._base_value)
            layout.addWidget(slider)
            #
            value_label = QtWidgets.QLabel(f"{slider.value() / self._base_value:.2f}")
            value_label.setAlignment(Qt.AlignRight)  # type: ignore[attr-defined]
            layout.addWidget(value_label)
            #
            slider.valueChanged.connect(self.onNewValue)
            slider.valueChanged.connect(
                lambda: value_label.setText(f"{slider.value() / self._base_value:.2f}")
            )
            layouts[title] = layout
            sliders[title] = slider

        self.slider_brightness = sliders["Brightness:"]
        self.slider_contrast = sliders["Contrast:"]
        del sliders

        # OK / Cancel buttons
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self._on_cancel)

        layout = QtWidgets.QVBoxLayout()  # type: ignore[assignment]
        layout.addLayout(layouts["Brightness:"])
        layout.addLayout(layouts["Contrast:"])
        layout.addWidget(btn_box)
        del layouts
        self.setLayout(layout)

    def _on_cancel(self):
        """Restore the original image before closing."""
        self.callback(self._original_qimage)
        self.reject()

    def onNewValue(self, _):
        brightness = self.slider_brightness.value() / self._base_value
        contrast = self.slider_contrast.value() / self._base_value

        img = self.img
        if brightness != 1:
            img = PIL.ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1:
            img = PIL.ImageEnhance.Contrast(img).enhance(contrast)

        qimage = QImage(
            img.tobytes(), img.width, img.height, img.width * 3, QImage.Format_RGB888
        )
        self.callback(qimage)
