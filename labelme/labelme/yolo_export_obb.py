"""
Export annotations to YOLO OBB (Oriented Bounding Box) format.

Label format per line:
  class_id  x1 y1  x2 y2  x3 y3  x4 y4
where (x1,y1)...(x4,y4) are the four corners of the rotated bounding box,
normalised to [0, 1] relative to image width/height.

Accepted annotation shapes:
  • rotation (2-point bbox + rotation angle in other_data) → rotated corners
  • polygon with exactly 4 points  → corners used directly
  • rectangle (2-point)            → converted to 4 axis-aligned corners
  • other polygons                  → skipped with a console warning
"""

import math
import os
import glob
import json
import shutil
import random
from PyQt5 import QtWidgets, QtCore
import yaml


def write_data_yaml(dataset_path, class_list):
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    data = {
        "train": os.path.join(dataset_path, "images", "train"),
        "val": os.path.join(dataset_path, "images", "val"),
        "nc": len(class_list),
        "names": class_list,
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return data_yaml_path


def _shape_to_obb_line(shape, class_list, img_w, img_h):
    """Convert a shape dict to a YOLO OBB label line, or return None to skip."""
    label = shape["label"]
    if label not in class_list:
        print(f"[OBB Export] Skipping shape with label '{label}' not in class_list")
        return None

    points = shape.get("points", [])
    shape_type = shape.get("shape_type", "polygon")
    class_id = class_list.index(label)

    if shape_type == "rotation" and len(points) >= 5:
        # New 5-point format: points[0..3] are the 4 actual rotated corners, points[4] is the handle
        corners = [(float(points[i][0]), float(points[i][1])) for i in range(4)]
    elif shape_type == "rotation" and len(points) >= 2:
        # Old format: points[0]=bbox TL, points[1]=bbox BR, other_data["rotation"]=degrees
        p0x, p0y = float(points[0][0]), float(points[0][1])
        p1x, p1y = float(points[1][0]), float(points[1][1])
        cx = (p0x + p1x) / 2
        cy = (p0y + p1y) / 2
        hw = abs(p1x - p0x) / 2
        hh = abs(p1y - p0y) / 2
        # rotation is stored at top-level in JSON (format_shape merges other_data into the dict)
        angle_deg = shape.get("rotation") or shape.get("other_data", {}).get("rotation") or 0.0
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        corners = []
        for lx, ly in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            corners.append((cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a))
    elif shape_type == "rectangle" and len(points) == 2:
        # Convert axis-aligned rectangle to 4 OBB corners
        x1, y1 = points[0]
        x2, y2 = points[1]
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    elif shape_type == "polygon" and len(points) == 4:
        corners = [(float(p[0]), float(p[1])) for p in points]
    elif shape_type == "polygon" and len(points) == 2:
        # Two-point polygon treated as rectangle
        x1, y1 = points[0]
        x2, y2 = points[1]
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    else:
        print(
            f"[OBB Export] Skipping shape '{label}' with {len(points)} points "
            f"(type={shape_type}). OBB requires exactly 4 corner points."
        )
        return None

    # Normalise and clamp
    norm = []
    for x, y in corners:
        nx = max(0.0, min(1.0, float(x) / img_w))
        ny = max(0.0, min(1.0, float(y) / img_h))
        norm.append(f"{nx:.6f} {ny:.6f}")

    return f"{class_id} " + " ".join(norm)


def export_to_yolo_obb(
    parent,
    dataset_path,
    lastOpenDir,
    uniqLabelList,
    get_split_dialog
):
    if not dataset_path:
        QtWidgets.QMessageBox.warning(
            parent,
            "Dataset Path Not Set",
            "Cannot export: No dataset path has been set.\n\n"
            "Please set the dataset path first using 'Select Project'.\n\n"
            "The dataset path is where your exported YOLO dataset will be saved."
        )
        parent.status("Export cancelled: Dataset path not set.")
        return

    if not lastOpenDir:
        QtWidgets.QMessageBox.warning(
            parent,
            "No Project Folder Opened",
            "Cannot export: No project folder with images is currently open.\n\n"
            "Please open a directory containing your images and annotations:\n"
            "• Menu: File > Open Dir"
        )
        parent.status("Export cancelled: No project folder opened.")
        return

    if not os.path.isdir(lastOpenDir):
        QtWidgets.QMessageBox.warning(parent, "Export Error", "Invalid project folder selected.")
        parent.status("Export cancelled: Invalid project folder.")
        return

    json_files = glob.glob(os.path.join(lastOpenDir, "*.json"))
    if not json_files:
        QtWidgets.QMessageBox.warning(
            parent,
            "No Annotations Found",
            "No JSON annotation files found in the project folder.\n\n"
            "Please annotate some images first before exporting."
        )
        parent.status("Export cancelled: No annotations found.")
        return

    has_annotations = False
    corrupt_files = []
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("shapes"):
                has_annotations = True
                break
        except json.JSONDecodeError as e:
            corrupt_files.append(os.path.basename(json_file))
        except Exception:
            pass

    if corrupt_files:
        QtWidgets.QMessageBox.warning(
            parent,
            "Corrupted Annotation Files",
            f"Found {len(corrupt_files)} corrupted JSON file(s):\n\n" +
            "\n".join(corrupt_files[:5]) +
            ("\n...and more" if len(corrupt_files) > 5 else "") +
            "\n\nThese files will be skipped."
        )

    if not has_annotations:
        QtWidgets.QMessageBox.warning(
            parent,
            "No Labeled Data",
            f"Found {len(json_files)} JSON file(s), but none contain any labeled objects.\n\n"
            "Please label at least some images before exporting."
        )
        parent.status("Export cancelled: No labeled objects found.")
        return

    # ── Merge class lists ──────────────────────────────────────────────────────
    src_labels_path = os.path.join(lastOpenDir, ".labels.json")
    dst_labels_path = os.path.join(dataset_path, ".labels.json")
    src_labels = []
    dst_labels = []
    if os.path.exists(src_labels_path):
        with open(src_labels_path, "r", encoding="utf-8") as f:
            src_labels = json.load(f).get("labels", [])
    if os.path.exists(dst_labels_path):
        with open(dst_labels_path, "r", encoding="utf-8") as f:
            dst_labels = json.load(f).get("labels", [])

    merged_labels = dst_labels.copy() if dst_labels else []
    new_labels = []
    for label in src_labels:
        if label not in merged_labels:
            merged_labels.append(label)
            new_labels.append(label)

    class_list = merged_labels if merged_labels else src_labels[:]
    for json_file in glob.glob(os.path.join(lastOpenDir, "*.json")):
        if os.path.basename(json_file) == ".labels.json":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                _d = json.load(f)
            for _s in _d.get("shapes", []):
                _lbl = _s.get("label", "")
                if _lbl and _lbl not in class_list:
                    class_list.append(_lbl)
        except Exception:
            pass

    if new_labels:
        QtWidgets.QMessageBox.information(
            parent,
            "New Labels Detected",
            f"New label(s) detected in source:\n\n{', '.join(new_labels)}\n\n"
            f"These will be added to the dataset.\n\n"
            f"Updated class list ({len(merged_labels)} total):\n{', '.join(merged_labels)}"
        )

    # ── Train/val split ────────────────────────────────────────────────────────
    dialog = get_split_dialog(parent)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        parent.status("Export cancelled.")
        return
    train_percent, val_percent = dialog.get_split()

    export_dir = dataset_path
    images_train_dir = os.path.join(export_dir, "images", "train")
    images_val_dir = os.path.join(export_dir, "images", "val")
    labels_train_dir = os.path.join(export_dir, "labels", "train")
    labels_val_dir = os.path.join(export_dir, "labels", "val")
    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Write .labels.json ─────────────────────────────────────────────────────
    for labels_json_path in [
        os.path.join(lastOpenDir, ".labels.json"),
        os.path.join(dataset_path, ".labels.json"),
    ]:
        try:
            with open(labels_json_path, "w", encoding="utf-8") as f:
                json.dump({"type": "obb", "labels": class_list}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                parent, "Failed to Update Labels",
                f"Could not update .labels.json:\n{str(e)}"
            )
            parent.status("Export cancelled: Failed to update labels.")
            return

    # ── Write data.yaml ────────────────────────────────────────────────────────
    try:
        write_data_yaml(dataset_path, class_list)
        parent.status(f"Updated data.yaml with {len(class_list)} classes")
    except Exception as e:
        QtWidgets.QMessageBox.warning(
            parent, "Failed to Update data.yaml", f"Could not update data.yaml:\n{str(e)}"
        )
        parent.status("Export cancelled: Failed to update data.yaml.")
        return

    # ── Convert annotations ────────────────────────────────────────────────────
    json_files = glob.glob(os.path.join(lastOpenDir, "*.json"))
    labeled = []
    verified_empty_images = []

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        shapes = data.get("shapes", [])
        is_verified_empty = data.get("verified_empty", False)

        if not shapes and not is_verified_empty:
            continue

        image_path = data.get("imagePath", "")
        candidates = [
            os.path.join(lastOpenDir, image_path),
            os.path.join(os.path.dirname(json_file), image_path),
            image_path,
        ]
        abs_image_path = None
        for candidate in candidates:
            if os.path.exists(candidate):
                abs_image_path = candidate
                break
        if not abs_image_path:
            continue

        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        if not img_w or not img_h:
            continue

        if is_verified_empty:
            verified_empty_images.append((abs_image_path, []))
            continue

        lines = []
        for shape in shapes:
            line = _shape_to_obb_line(shape, class_list, img_w, img_h)
            if line is not None:
                lines.append(line)

        if not lines:
            continue

        labeled.append((abs_image_path, lines))

    all_images = labeled + verified_empty_images
    total_verified_empty = len(verified_empty_images)

    # ── Shuffle and split ──────────────────────────────────────────────────────
    random.shuffle(all_images)
    n_train = int(len(all_images) * train_percent / 100)
    train_set = all_images[:n_train]
    val_set = all_images[n_train:]

    total = len(train_set) + len(val_set)
    progress = QtWidgets.QProgressDialog("Exporting OBB dataset...", "Cancel", 0, total, parent)
    progress.setWindowTitle("Export Progress")
    progress.setWindowModality(QtCore.Qt.WindowModal)
    progress.show()
    QtWidgets.QApplication.processEvents()

    def image_split(image_filename):
        if os.path.exists(os.path.join(images_train_dir, image_filename)):
            return 'train'
        if os.path.exists(os.path.join(images_val_dir, image_filename)):
            return 'val'
        return None

    def write_set(image_set, default_split):
        for i, (abs_image_path, lines) in enumerate(image_set):
            image_filename = os.path.basename(abs_image_path)
            label_filename = os.path.splitext(image_filename)[0] + ".txt"
            split = image_split(image_filename) or default_split
            target_label_path = os.path.join(
                labels_train_dir if split == 'train' else labels_val_dir, label_filename
            )
            with open(target_label_path, "w") as f:
                f.write("\n".join(lines))
            if image_split(image_filename) is None:
                target_image_path = os.path.join(
                    images_train_dir if split == 'train' else images_val_dir, image_filename
                )
                shutil.copy2(abs_image_path, target_image_path)
            yield i

    count = 0
    for _ in write_set(train_set, 'train'):
        count += 1
        progress.setValue(count)
        QtWidgets.QApplication.processEvents()
        if progress.wasCanceled():
            parent.status("Export cancelled.")
            return

    for _ in write_set(val_set, 'val'):
        count += 1
        progress.setValue(count)
        QtWidgets.QApplication.processEvents()
        if progress.wasCanceled():
            parent.status("Export cancelled.")
            return

    progress.setValue(total)

    status_msg = f"Exported {len(train_set)} train and {len(val_set)} val images/labels to '{export_dir}'."
    if total_verified_empty > 0:
        status_msg += f" (Includes {total_verified_empty} verified empty images)"
    parent.status(status_msg)

    if hasattr(parent, 'projectLabelList'):
        parent.projectLabelList.clear()
        for label in class_list:
            parent.projectLabelList.addItem(QtWidgets.QListWidgetItem(label))


def set_dataset_path_obb(parent, current_path, settings):
    path = QtWidgets.QFileDialog.getExistingDirectory(
        parent,
        parent.tr("Select Dataset Directory"),
        current_path or ".",
        QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks,
    )
    if path:
        settings.setValue("dataset_path", path)
        parent.status(f"Dataset path set to: {path}")
    return path
