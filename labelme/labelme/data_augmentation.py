# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from pathlib import Path
import random
import re
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class AugmentationWorker(QThread):
    """Worker thread for performing data augmentation"""
    progress_signal = pyqtSignal(int, str)  # (progress, message)
    finished_signal = pyqtSignal(str)  # (summary message)
    error_signal = pyqtSignal(str)  # (error message)
    
    def __init__(self, dataset_path, task_type, multiplier, augmentations, intensities):
        super().__init__()
        self.dataset_path = dataset_path
        self.task_type = task_type
        self.multiplier = multiplier
        self.augmentations = augmentations  # List of selected augmentation names
        self.intensities = intensities  # Dict of augmentation name -> intensity value
        self._should_stop = False
        
    def run(self):
        try:
            self.perform_augmentation()
        except Exception as e:
            self.error_signal.emit(f"Augmentation failed: {str(e)}")
    
    def stop(self):
        self._should_stop = True
    
    def perform_augmentation(self):
        """Main augmentation logic"""
        # Find all images in dataset - ONLY in train folder
        images_dir = os.path.join(self.dataset_path, "images", "train")
        labels_dir = os.path.join(self.dataset_path, "labels", "train")
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            self.error_signal.emit("Train images or labels directory not found in dataset.\nExpected: images/train and labels/train")
            return
        
        # Get all image files in train folder
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        all_images = []
        augmented_images = []
        
        for ext in image_extensions:
            for img_path in Path(images_dir).glob(f"*{ext}"):
                # Skip images that have been already augmented (contain "_aug_" in filename)
                if "_aug_" not in img_path.stem:
                    all_images.append(img_path)
                else:
                    augmented_images.append(img_path)
        
        if not all_images:
            self.error_signal.emit("No original images found in train folder (only augmented images exist)")
            return
        
        # Build set of originals that already have augmented variants and skip them.
        # Example: car_01.jpg has augmented files car_01_aug_1.jpg, car_01_aug_2.jpg
        aug_pattern = re.compile(r"^(?P<base>.+)_aug_\d+$")
        already_augmented_bases = set()
        for aug_img in augmented_images:
            match = aug_pattern.match(aug_img.stem)
            if match:
                already_augmented_bases.add(match.group("base"))

        eligible_images = [img for img in all_images if img.stem not in already_augmented_bases]

        original_count = len(all_images)
        skipped_count = original_count - len(eligible_images)

        # We only augment originals that have never been augmented before.
        # For each eligible original, generate (multiplier - 1) new images.
        augmentations_needed = len(eligible_images) * max(self.multiplier - 1, 0)
        
        status_msg = f"Found {original_count} original images in train folder"
        if augmented_images:
            status_msg += f" and {len(augmented_images)} augmented images"
        if skipped_count > 0:
            status_msg += f". Skipping {skipped_count} originals that were already augmented"
        status_msg += f". Applying {self.multiplier}x on {len(eligible_images)} eligible originals"
        status_msg += ". Validation folder unchanged"
        
        self.progress_signal.emit(0, status_msg)

        if augmentations_needed == 0:
            self.finished_signal.emit(
                "No new augmentation needed.\n"
                "All original train images already have augmented versions."
            )
            return
        
        # Perform augmentations: enforce multiplier per eligible original image
        import time as _time
        augmented_count = 0
        task_index = 0
        _aug_start = _time.time()
        eligible_images_shuffled = eligible_images[:]
        random.shuffle(eligible_images_shuffled)

        for source_image_path in eligible_images_shuffled:
            for aug_number in range(1, self.multiplier):
                if self._should_stop:
                    self.finished_signal.emit("Augmentation cancelled by user")
                    return

                task_index += 1
            
                # Load image
                image = cv2.imread(str(source_image_path))
                if image is None:
                    continue

                # Load corresponding label
                label_path = self.get_label_path(source_image_path, labels_dir)
                if not os.path.exists(label_path):
                    continue

                # Load label data
                with open(label_path, 'r') as f:
                    label_lines = f.readlines()

                # Randomly select augmentations to apply
                selected_augs = random.sample(
                    self.augmentations,
                    k=min(random.randint(1, 3), len(self.augmentations))
                )

                # Apply augmentations
                augmented_image = image.copy()
                augmented_labels = [line.strip() for line in label_lines]

                for aug_name in selected_augs:
                    augmented_image, augmented_labels = self.apply_augmentation(
                        aug_name, augmented_image, augmented_labels
                    )

                # Save augmented image and label (suffix index per original image)
                aug_suffix = f"_aug_{aug_number}"
                new_image_path = self.get_augmented_path(source_image_path, aug_suffix)
                new_label_path = self.get_augmented_path(Path(label_path), aug_suffix, extension='.txt')

                cv2.imwrite(str(new_image_path), augmented_image)

                with open(new_label_path, 'w') as f:
                    for label_line in augmented_labels:
                        f.write(label_line + '\n')

                augmented_count += 1
                progress = int((task_index / augmentations_needed) * 100)
                elapsed = _time.time() - _aug_start
                if augmented_count > 0 and augmented_count < augmentations_needed:
                    eta_secs = int(elapsed / augmented_count * (augmentations_needed - augmented_count))
                    eta_min, eta_s = divmod(eta_secs, 60)
                    eta_str = f"  |  ETA {eta_min:02d}:{eta_s:02d}"
                else:
                    eta_str = ""
                self.progress_signal.emit(
                    progress,
                    f"Augmented {augmented_count}/{augmentations_needed} train images{eta_str}"
                )
        
        final_augmented_total = len(augmented_images) + augmented_count
        summary = f"Augmentation complete!\n" \
             f"Train folder - Original images: {original_count}\n" \
             f"Train folder - Existing augmented images: {len(augmented_images)}\n" \
             f"Train folder - Newly augmented images: {augmented_count}\n" \
             f"Train folder - Total augmented images: {final_augmented_total}\n" \
             f"Train folder - Total images: {original_count + final_augmented_total}\n" \
                 f"\nValidation folder: unchanged"
        self.finished_signal.emit(summary)
    
    def apply_augmentation(self, aug_name, image, labels):
        """Apply a specific augmentation to image and labels"""
        _ = image.shape[:2]
        
        if aug_name == "flip":
            image = cv2.flip(image, 1)  # Horizontal flip
            labels = self.flip_labels(labels, horizontal=True)
        
        elif aug_name == "reverse":
            image = cv2.flip(image, 0)  # Vertical flip
            labels = self.flip_labels(labels, horizontal=False)
        
        elif aug_name == "rotate90":
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            labels = self.rotate_labels_90(labels)
        
        elif aug_name == "grayscale":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        elif aug_name == "blur":
            intensity = self.intensities.get(aug_name, 0.5)
            kernel_size = int(1 + intensity * 20)
            if kernel_size % 2 == 0:
                kernel_size += 1
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        elif aug_name == "brightness":
            intensity = self.intensities.get(aug_name, 0.2)
            # Randomly choose to brighten or darken
            factor = random.choice([1 + intensity, 1 - intensity])
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        elif aug_name == "saturation":
            intensity = self.intensities.get(aug_name, 0.2)
            # Randomly choose to increase or decrease saturation
            factor = random.choice([1 + intensity, 1 - intensity])
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        elif aug_name == "noise":
            intensity = self.intensities.get(aug_name, 0.1)
            noise = np.random.randn(*image.shape) * (intensity * 50)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image, labels
    
    def flip_labels(self, labels, horizontal=True):
        """Flip YOLO format labels"""
        flipped_labels = []
        for label in labels:
            parts = label.split()
            if len(parts) < 5:
                continue
            
            class_id = parts[0]
            
            if self.task_type == "bbox":
                # Bounding box format: class x_center y_center width height
                x_center, y_center, width, height = map(float, parts[1:5])
                
                if horizontal:
                    x_center = 1.0 - x_center
                else:
                    y_center = 1.0 - y_center
                
                flipped_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
            
            else:  # Segmentation
                # Polygon format: class x1 y1 x2 y2 x3 y3 ...
                coords = list(map(float, parts[1:]))
                flipped_coords = []
                
                for i in range(0, len(coords), 2):
                    x, y = coords[i], coords[i+1]
                    if horizontal:
                        x = 1.0 - x
                    else:
                        y = 1.0 - y
                    flipped_coords.extend([x, y])
                
                coords_str = ' '.join(map(str, flipped_coords))
                flipped_labels.append(f"{class_id} {coords_str}")
        
        return flipped_labels
    
    def rotate_labels_90(self, labels):
        """Rotate YOLO format labels 90 degrees clockwise"""
        rotated_labels = []
        for label in labels:
            parts = label.split()
            if len(parts) < 5:
                continue
            
            class_id = parts[0]
            
            if self.task_type == "bbox":
                # Bounding box format
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # Rotate 90 degrees clockwise: (x, y) -> (1-y, x)
                new_x = 1.0 - y_center
                new_y = x_center
                new_width = height
                new_height = width
                
                rotated_labels.append(f"{class_id} {new_x} {new_y} {new_width} {new_height}")
            
            else:  # Segmentation
                coords = list(map(float, parts[1:]))
                rotated_coords = []
                
                for i in range(0, len(coords), 2):
                    x, y = coords[i], coords[i+1]
                    # Rotate 90 degrees clockwise
                    new_x = 1.0 - y
                    new_y = x
                    rotated_coords.extend([new_x, new_y])
                
                coords_str = ' '.join(map(str, rotated_coords))
                rotated_labels.append(f"{class_id} {coords_str}")
        
        return rotated_labels
    
    def get_label_path(self, image_path, labels_dir):
        """Get corresponding label path for an image in train folder"""
        # Since we're now working directly in images/train and labels/train,
        # we just need to match the filename
        label_path = Path(labels_dir) / (image_path.stem + '.txt')
        return str(label_path)
    
    def get_augmented_path(self, original_path, suffix, extension=None):
        """Generate augmented file path"""
        original_path = Path(original_path)
        if extension is None:
            extension = original_path.suffix
        new_name = original_path.stem + suffix + extension
        return original_path.parent / new_name


class DataAugmentationDialog(QtWidgets.QDialog):
    """Dialog for configuring and running data augmentation"""
    
    def __init__(self, parent, dataset_path, task_type):
        super().__init__(parent)
        self.dataset_path = dataset_path
        self.task_type = task_type
        self.worker = None
        self.selected_augmentations = set()
        self.intensities = {}
        
        self.setWindowTitle("Data Augmentation")
        self.setStyleSheet("color: #ffffff;")
        self.setModal(True)

        self.setup_ui()

        from labelme.utils import fit_dialog
        fit_dialog(self, 780, 700)
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QtWidgets.QLabel("Dataset Augmentation")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #C1FFB5;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Multiplier selection
        multiplier_group = QtWidgets.QGroupBox("Augmentation Multiplier")
        multiplier_layout = QtWidgets.QVBoxLayout()

        spin_row = QtWidgets.QHBoxLayout()
        spin_row.addWidget(QtWidgets.QLabel("Total copies per image:"))
        self.multiplier_spin = QtWidgets.QSpinBox()
        self.multiplier_spin.setMinimum(2)
        self.multiplier_spin.setMaximum(10)
        self.multiplier_spin.setValue(3)
        self.multiplier_spin.setSuffix("x")
        spin_row.addWidget(self.multiplier_spin)
        spin_row.addStretch()
        multiplier_layout.addLayout(spin_row)

        self.multiplier_info_label = QtWidgets.QLabel()
        self.multiplier_info_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        self._update_multiplier_info(self.multiplier_spin.value())
        multiplier_layout.addWidget(self.multiplier_info_label)
        self.multiplier_spin.valueChanged.connect(self._update_multiplier_info)

        multiplier_group.setLayout(multiplier_layout)
        layout.addWidget(multiplier_group)
        
        # Augmentation selection — 2-column grid inside a scroll area
        self.aug_buttons = {}
        COLS = 2
        all_augs = [
            ("flip",       "Flip Horizontal", "flip",       False),
            ("reverse",    "Flip Vertical",   "reverse",    False),
            ("rotate90",   "Rotate 90°",      "rotate",     False),
            ("grayscale",  "Grayscale",        "grayscale",  False),
            ("blur",       "Blur",             "blur",       True),
            ("brightness", "Brightness",       "brightness", True),
            ("saturation", "Saturation",       "saturation", True),
            ("noise",      "Noise",            "noise",      True),
        ]

        aug_grid_widget = QtWidgets.QWidget()
        aug_layout = QtWidgets.QGridLayout(aug_grid_widget)
        aug_layout.setSpacing(12)
        aug_layout.setContentsMargins(8, 8, 8, 8)

        for idx, (aug_id, aug_name, icon_name, has_intensity) in enumerate(all_augs):
            card = self.create_aug_button(aug_id, aug_name, icon_name,
                                          has_intensity=has_intensity)
            self.aug_buttons[aug_id] = card
            aug_layout.addWidget(card, idx // COLS, idx % COLS)

        for c in range(COLS):
            aug_layout.setColumnStretch(c, 1)

        aug_scroll = QtWidgets.QScrollArea()
        aug_scroll.setWidgetResizable(True)
        aug_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        aug_scroll.setWidget(aug_grid_widget)

        aug_group = QtWidgets.QGroupBox("Select Augmentations")
        aug_group_layout = QtWidgets.QVBoxLayout(aug_group)
        aug_group_layout.setContentsMargins(4, 4, 4, 4)
        aug_group_layout.addWidget(aug_scroll)
        layout.addWidget(aug_group, stretch=1)
        
        # Progress section
        progress_group = QtWidgets.QGroupBox("Progress")
        progress_layout = QtWidgets.QVBoxLayout()
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QtWidgets.QLabel("Ready to augment")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.start_btn = QtWidgets.QPushButton("Start Augmentation")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.start_btn.clicked.connect(self.start_augmentation)
        button_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.cancel_btn.clicked.connect(self.cancel_augmentation)
        button_layout.addWidget(self.cancel_btn)
        
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)

        self.remove_aug_btn = QtWidgets.QPushButton("Remove Augmented Images")
        self.remove_aug_btn.setToolTip("Delete all _aug_ images and labels from the train folder")
        self.remove_aug_btn.setStyleSheet("""
            QPushButton {
                background-color: #b71c1c;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.remove_aug_btn.clicked.connect(self.remove_augmented_images)
        button_layout.addWidget(self.remove_aug_btn)

        layout.addLayout(button_layout)
    
    def _update_multiplier_info(self, value):
        """Update the explanatory label beneath the multiplier spinner."""
        new_copies = value - 1
        noun = "copy" if new_copies == 1 else "copies"
        self.multiplier_info_label.setText(
            f"→ Creates {new_copies} new augmented {noun} per original image"
        )

    def create_aug_button(self, aug_id, aug_name, icon_name, has_intensity=False):
        """Create an augmentation card: toggle button on top, optional slider below."""
        # Card frame
        container = QtWidgets.QFrame()
        container.setFrameShape(QtWidgets.QFrame.StyledPanel)
        container.setStyleSheet("""
            QFrame {
                background: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 5px;
            }
        """)
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(10, 5, 10, 5)
        container_layout.setSpacing(8)

        # Toggle button (image + name area)
        btn = QtWidgets.QPushButton()
        btn.setCheckable(True)
        btn.setMinimumHeight(90)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        btn.setStyleSheet("""
            QPushButton {
                color: #222222;
                border: 2px solid #cccccc;
                border-radius: 6px;
                background: #ffffff;
                font-weight: bold;
                font-size: 13px;
                padding: 6px;
            }
            QPushButton:checked {
                border: 2px solid #4CAF50;
                background: #e8f5e9;
                color: #1b5e20;
            }
            QPushButton:hover {
                border-color: #888adf;
                background: #f5f5ff;
            }
        """)

        try:
            from labelme.utils import newIcon
            btn.setIcon(newIcon(icon_name))
            btn.setIconSize(QtCore.QSize(48, 32))
        except Exception:
            pass

        btn.setText(aug_name)
        btn.setToolButtonStyle = lambda *_: None  # no-op — it's a QPushButton
        btn.clicked.connect(lambda checked, aid=aug_id: self.toggle_augmentation(aid, checked))
        container_layout.addWidget(btn)

        # Intensity slider — always in its own row below the button
        if has_intensity:
            slider_row = QtWidgets.QHBoxLayout()
            slider_row.setSpacing(6)

            intensity_lbl = QtWidgets.QLabel("Intensity:")
            intensity_lbl.setStyleSheet("color: #555555; font-size: 11px;")
            slider_row.addWidget(intensity_lbl)

            slider = QtWidgets.QSlider(Qt.Horizontal)
            slider.setMinimum(1)
            slider.setMaximum(100)
            slider.setValue(20)
            slider.setEnabled(False)
            slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 5px;
                    background: #dddddd;
                    border-radius: 2px;
                }
                QSlider::handle:horizontal {
                    background: #ffffff;
                    border: 2px solid #aaaaaa;
                    width: 14px;
                    height: 14px;
                    margin: -5px 0;
                    border-radius: 7px;
                }
                QSlider::handle:horizontal:hover {
                    border-color: #555555;
                }
                QSlider::sub-page:horizontal {
                    background: #888888;
                    border-radius: 2px;
                }
                QSlider::groove:horizontal:disabled {
                    background: #eeeeee;
                }
                QSlider::sub-page:horizontal:disabled {
                    background: #cccccc;
                }
                QSlider::handle:horizontal:disabled {
                    background: #f0f0f0;
                    border-color: #cccccc;
                }
            """)
            slider.valueChanged.connect(lambda v, aid=aug_id: self.update_intensity(aid, v))
            slider_row.addWidget(slider)

            value_label = QtWidgets.QLabel("0.20")
            value_label.setFixedWidth(36)
            value_label.setStyleSheet("color: #333333; font-size: 11px;")
            slider_row.addWidget(value_label)

            container_layout.addLayout(slider_row)

            btn.intensity_slider = slider
            btn.intensity_label  = value_label
        else:
            # Spacer so geometry-only cards have the same total height as
            # intensity cards — keeps the grid rows uniform.
            container_layout.addSpacing(28)

        return container
    
    def toggle_augmentation(self, aug_id, checked):
        """Toggle augmentation selection"""
        if checked:
            self.selected_augmentations.add(aug_id)
            # Enable intensity slider if exists
            btn_widget = self.aug_buttons[aug_id]
            btn = btn_widget.findChild(QtWidgets.QPushButton)
            if hasattr(btn, 'intensity_slider'):
                btn.intensity_slider.setEnabled(True)
                self.update_intensity(aug_id, btn.intensity_slider.value())
        else:
            self.selected_augmentations.discard(aug_id)
            # Disable intensity slider if exists
            btn_widget = self.aug_buttons[aug_id]
            btn = btn_widget.findChild(QtWidgets.QPushButton)
            if hasattr(btn, 'intensity_slider'):
                btn.intensity_slider.setEnabled(False)
    
    def update_intensity(self, aug_id, value):
        """Update intensity value"""
        intensity = value / 100.0
        self.intensities[aug_id] = intensity
        
        # Update label
        btn_widget = self.aug_buttons[aug_id]
        btn = btn_widget.findChild(QtWidgets.QPushButton)
        if hasattr(btn, 'intensity_label'):
            btn.intensity_label.setText(f"{intensity:.2f}")
    
    def start_augmentation(self):
        """Start the augmentation process"""
        if not self.selected_augmentations:
            QtWidgets.QMessageBox.warning(self, "No Augmentations", 
                                         "Please select at least one augmentation to apply.")
            return
        
        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting augmentation...")
        
        # Create worker thread
        multiplier = self.multiplier_spin.value()
        self.worker = AugmentationWorker(
            self.dataset_path,
            self.task_type,
            multiplier,
            list(self.selected_augmentations),
            self.intensities
        )
        
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.augmentation_finished)
        self.worker.error_signal.connect(self.augmentation_error)
        
        self.worker.start()
    
    def cancel_augmentation(self):
        """Cancel ongoing augmentation"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("Cancelling...")
        else:
            self.reject()
    
    def update_progress(self, progress, message):
        """Update progress bar and status"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def augmentation_finished(self, summary):
        """Handle augmentation completion"""
        self.start_btn.setEnabled(True)
        self.status_label.setText("Augmentation complete!")
        QtWidgets.QMessageBox.information(self, "Success", summary)
    
    def augmentation_error(self, error_msg):
        """Handle augmentation error"""
        self.start_btn.setEnabled(True)
        self.status_label.setText("Error occurred")
        QtWidgets.QMessageBox.critical(self, "Error", error_msg)

    def remove_augmented_images(self):
        """Delete all _aug_ images and label files from the train folder."""
        images_dir = os.path.join(self.dataset_path, "images", "train")
        labels_dir = os.path.join(self.dataset_path, "labels", "train")

        aug_images = [
            f for f in Path(images_dir).glob("*_aug_*")
        ] if os.path.exists(images_dir) else []
        aug_labels = [
            f for f in Path(labels_dir).glob("*_aug_*")
        ] if os.path.exists(labels_dir) else []

        total = len(aug_images) + len(aug_labels)
        if total == 0:
            QtWidgets.QMessageBox.information(self, "No Augmented Files", "No augmented files found in the train folder.")
            return

        answer = QtWidgets.QMessageBox.warning(
            self,
            "Remove Augmented Images",
            f"This will permanently delete {len(aug_images)} augmented images "
            f"and {len(aug_labels)} augmented label files from the train folder.\n\nProceed?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        removed = 0
        for f in aug_images + aug_labels:
            try:
                f.unlink()
                removed += 1
            except Exception:
                pass

        self.status_label.setText(f"Removed {removed} augmented files.")
        QtWidgets.QMessageBox.information(self, "Done", f"Removed {removed} augmented files.")
