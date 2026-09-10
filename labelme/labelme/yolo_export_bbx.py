import os
import glob
import json
import random
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import yaml

from labelme.yolo_export_common import dataset_dirs
from labelme.yolo_export_common import export_image
from labelme.yolo_export_common import remove_from_dataset
from labelme.yolo_export_common import resolve_image_path
from labelme.yolo_export_common import shape_outline

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

def export_to_yolov5_bbx(
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
            "Please set the dataset path first using:\n"
            "• Menu: YOLO > Set Dataset Path (Bounding Box)\n\n"
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
            "• Menu: File > Open Dir\n\n"
            "The folder should contain your images and their corresponding JSON annotation files."
        )
        parent.status("Export cancelled: No project folder opened.")
        return
    
    # Validate that the project folder exists and has content
    if not os.path.isdir(lastOpenDir):
        QtWidgets.QMessageBox.warning(
            parent,
            "Export Error",
            "Invalid project folder selected."
        )
        parent.status("Export cancelled: Invalid project folder.")
        return
    
    # Count JSON files BEFORE showing split dialog
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
    
    # Check if any JSON files have actual shapes
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
            print(f"[WARNING] Corrupted JSON file: {json_file} - {str(e)}")
            continue
        except Exception as e:
            print(f"[WARNING] Failed to read JSON file {json_file}: {str(e)}")
            continue
    
    if corrupt_files:
        QtWidgets.QMessageBox.warning(
            parent,
            "Corrupted Annotation Files",
            f"Found {len(corrupt_files)} corrupted JSON file(s):\n\n" +
            "\n".join(corrupt_files[:5]) +
            ("\n...and more" if len(corrupt_files) > 5 else "") +
            "\n\nThese files will be skipped. You may want to fix or remove them."
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

    # Merge labels: keep existing dataset labels, append new ones from source
    merged_labels = dst_labels.copy() if dst_labels else []
    new_labels = []
    
    for label in src_labels:
        if label not in merged_labels:
            merged_labels.append(label)
            new_labels.append(label)
    
    # Use merged labels as the class list, but also scan JSON files to pick up
    # any labels that were annotated but never added to .labels.json
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
    
    # If there are new labels, inform user
    if new_labels:
        msg = f"New label(s) detected in source:\n\n{', '.join(new_labels)}\n\n"
        msg += f"These will be added to the dataset.\n\n"
        msg += f"Updated class list ({len(merged_labels)} total):\n{', '.join(merged_labels)}"
        
        QtWidgets.QMessageBox.information(
            parent,
            "New Labels Detected",
            msg,
            QtWidgets.QMessageBox.Ok
        )

    # Show train/val split dialog
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

    # Update .labels.json in PROJECT folder with merged class list
    project_label_json_path = os.path.join(lastOpenDir, ".labels.json")
    try:
        with open(project_label_json_path, "w", encoding="utf-8") as f:
            json.dump({"type": "bbox", "labels": class_list}, f, ensure_ascii=False, indent=2)
        if new_labels:
            parent.status(f"Updated project labels: added {len(new_labels)} new label(s)")
    except Exception as e:
        print(f"Warning: Could not update project .labels.json: {e}")

    # Update .labels.json in DATASET folder with merged class list
    label_json_path = os.path.join(dataset_path, ".labels.json")
    try:
        with open(label_json_path, "w", encoding="utf-8") as f:
            json.dump({"type": "bbox", "labels": class_list}, f, ensure_ascii=False, indent=2)
        if new_labels:
            parent.status(f"Updated dataset labels: added {len(new_labels)} new label(s)")
    except Exception as e:
        QtWidgets.QMessageBox.warning(
            parent,
            "Failed to Update Labels",
            f"Could not update .labels.json in dataset:\n{str(e)}"
        )
        parent.status("Export cancelled: Failed to update labels.")
        return
    
    # Update data.yaml with merged/final class list
    try:
        data_yaml_path = write_data_yaml(dataset_path, class_list)
        parent.status(f"Updated data.yaml with {len(class_list)} classes")
    except Exception as e:
        QtWidgets.QMessageBox.warning(
            parent,
            "Failed to Update data.yaml",
            f"Could not update data.yaml:\n{str(e)}"
        )
        parent.status("Export cancelled: Failed to update data.yaml.")
        return

    # Find all JSON label files in the current directory
    json_files = glob.glob(os.path.join(lastOpenDir, "*.json"))
    labeled = []
    verified_empty_images = []  # Track images marked as verified empty
    unlabeled_images = []  # Images with no shapes (stale copies get removed)
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        shapes = data.get("shapes", [])

        # Check if image is marked as verified empty
        is_verified_empty = data.get("verified_empty", False)

        if not shapes and not is_verified_empty:
            # Nothing to export, but remember it so a stale copy from an
            # earlier export (shapes since deleted) is dropped from the dataset
            unlabeled_images.append(resolve_image_path(data, json_file, lastOpenDir))
            continue

        abs_image_path = resolve_image_path(data, json_file, lastOpenDir)
        if not abs_image_path:
            continue  # skip if image not found

        img_w = data.get("imageWidth", None)
        img_h = data.get("imageHeight", None)
        if img_w is None or img_h is None:
            continue  # skip if no image size

        # If verified empty, add to separate list with empty labels
        if is_verified_empty:
            verified_empty_images.append((abs_image_path, []))
            continue

        # Prepare YOLOv5 label lines
        lines = []
        for shape in shapes:
            label = shape["label"]
            if label not in class_list:
                print(f"Skipping shape with label {label} not in class_list")
                continue
            # Bounding box of the shape's outline. Using the raw points would
            # include a rotated box's handle point or a circle's edge point.
            outline = shape_outline(shape)
            if outline is None:
                continue
            xs = [p[0] for p in outline]
            ys = [p[1] for p in outline]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            class_id = class_list.index(label)
            cx = max(0.0, min(1.0, (x_min + x_max) / 2 / img_w))
            cy = max(0.0, min(1.0, (y_min + y_max) / 2 / img_h))
            w  = max(0.0, min(1.0, (x_max - x_min) / img_w))
            h  = max(0.0, min(1.0, (y_max - y_min) / img_h))
            line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            lines.append(line)

        if not lines:
            continue  # skip if no valid shapes

        labeled.append((abs_image_path, lines))
    
    # Combine labeled images with verified empty images
    all_images = labeled + verified_empty_images
    total_verified_empty = len(verified_empty_images)

    # Shuffle and split
    random.shuffle(all_images)
    n_train = int(len(all_images) * train_percent / 100)
    train_set = all_images[:n_train]
    val_set = all_images[n_train:]

    # Progress dialog
    total = len(train_set) + len(val_set)
    progress = QtWidgets.QProgressDialog("Exporting to YOLOv5...", "Cancel", 0, total, parent)
    progress.setWindowTitle("Export Progress")
    progress.setWindowModality(QtCore.Qt.WindowModal)
    progress.show()
    QtWidgets.QApplication.processEvents()

    # Images already in the dataset (matched by content) keep their split and
    # get their label updated; new images are copied in under a free name
    dirs = dataset_dirs(export_dir)
    for i, (abs_image_path, lines) in enumerate(train_set + val_set):
        default_split = "train" if i < len(train_set) else "val"
        export_image(abs_image_path, lines, default_split, dirs)

        progress.setValue(i)
        QtWidgets.QApplication.processEvents()
        if progress.wasCanceled():
            parent.status("Export cancelled.")
            return

    # Drop dataset copies of images whose shapes were all deleted
    removed = sum(remove_from_dataset(p, dirs) for p in unlabeled_images if p)

    progress.setValue(total)
    
    # Build status message with verified empty count
    status_msg = f"Exported {len(train_set)} train and {len(val_set)} val images/labels to '{export_dir}'."
    if total_verified_empty > 0:
        status_msg += f" (Includes {total_verified_empty} verified empty images)"
    parent.status(status_msg)
    if removed:
        QtWidgets.QMessageBox.information(
            parent,
            "Stale Images Removed",
            f"Removed {removed} image(s) from the dataset because all of their "
            "shapes were deleted since the last export.",
        )
    
    # Update Project Labels UI with merged label list (always update to ensure sync)
    if hasattr(parent, 'projectLabelList'):
        # Clear and rebuild the project label list with merged labels
        parent.projectLabelList.clear()
        for label in class_list:
            item = QtWidgets.QListWidgetItem(label)
            parent.projectLabelList.addItem(item)
        if new_labels:
            parent.status(f"Updated project labels in UI with {len(class_list)} labels (added {len(new_labels)} new)")
        else:
            parent.status(f"Synchronized project labels in UI with {len(class_list)} labels")

def set_dataset_path_bbx(parent, current_path, settings):
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
