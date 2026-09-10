"""Helpers shared by the YOLO segmentation / bounding-box / OBB exporters."""

import filecmp
import math
import os
import shutil

_CIRCLE_SEGMENTS = 32


def rotation_corners(shape):
    """Return the 4 corners of a labelme "rotation" shape dict, or None.

    New format: points[0..3] are the corners (points[4] is the rotation
    handle). Old format: points[0..1] are the unrotated box and the angle is
    in "rotation" (top level in saved JSON) or other_data["rotation"].
    """
    points = shape.get("points", [])
    if len(points) >= 4:
        return [(float(x), float(y)) for x, y in points[:4]]
    if len(points) < 2:
        return None
    (x0, y0), (x1, y1) = points[0], points[1]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = abs(x1 - x0) / 2, abs(y1 - y0) / 2
    angle_deg = (
        shape.get("rotation")
        or (shape.get("other_data") or {}).get("rotation")
        or 0.0
    )
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    return [
        (cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a)
        for lx, ly in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    ]


def shape_outline(shape):
    """Return the outline of an area shape as a list of (x, y), or None.

    Rectangles, rotated boxes and circles are converted to polygons. Lines,
    points and degenerate polygons have no area and return None.
    """
    shape_type = shape.get("shape_type") or "polygon"
    points = [(float(p[0]), float(p[1])) for p in shape.get("points", [])]

    if shape_type == "polygon" and len(points) >= 3:
        return points
    if shape_type in ("rectangle", "polygon") and len(points) == 2:
        # 2-point polygons are boxes saved by older code paths
        (x0, y0), (x1, y1) = points
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if shape_type == "rotation":
        return rotation_corners(shape)
    if shape_type == "circle" and len(points) == 2:
        (cx, cy), (ex, ey) = points
        r = math.hypot(ex - cx, ey - cy)
        if r == 0:
            return None
        return [
            (
                cx + r * math.cos(2 * math.pi * i / _CIRCLE_SEGMENTS),
                cy + r * math.sin(2 * math.pi * i / _CIRCLE_SEGMENTS),
            )
            for i in range(_CIRCLE_SEGMENTS)
        ]
    return None


def resolve_image_path(data, json_file, project_dir):
    """Return the path of the image a labelme JSON refers to, or None."""
    image_path = data.get("imagePath", "")
    if not image_path:
        return None
    for candidate in (
        os.path.join(project_dir, image_path),
        os.path.join(os.path.dirname(json_file), image_path),
        image_path,  # in case it's absolute
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def dataset_dirs(dataset_path):
    """Return {"train": (images_dir, labels_dir), "val": (...)}."""
    return {
        split: (
            os.path.join(dataset_path, "images", split),
            os.path.join(dataset_path, "labels", split),
        )
        for split in ("train", "val")
    }


def _candidate_names(filename):
    stem, ext = os.path.splitext(filename)
    yield filename
    i = 1
    while True:
        yield f"{stem}_{i}{ext}"
        i += 1


def find_dataset_copy(abs_image_path, dirs):
    """Locate this image in the dataset.

    Images are matched by content, not only by filename, so two different
    source images that share a name (e.g. img_001.jpg from two project
    folders) never overwrite each other's labels.

    Returns (split, filename). split is "train"/"val" when an identical copy
    is already in the dataset; otherwise it is None and filename is a free
    name to copy the image to.
    """
    for name in _candidate_names(os.path.basename(abs_image_path)):
        taken = False
        for split, (images_dir, _) in dirs.items():
            existing = os.path.join(images_dir, name)
            if os.path.exists(existing):
                taken = True
                if filecmp.cmp(abs_image_path, existing, shallow=False):
                    return split, name
        if not taken:
            return None, name


def export_image(abs_image_path, lines, default_split, dirs):
    """Copy an image into the dataset if needed and write its label file.

    An image already in the dataset keeps its current split, so re-exporting
    doesn't move images between train and val. Returns (split, filename).
    """
    split, name = find_dataset_copy(abs_image_path, dirs)
    if split is None:
        split = default_split
        shutil.copy2(abs_image_path, os.path.join(dirs[split][0], name))
    label_path = os.path.join(dirs[split][1], os.path.splitext(name)[0] + ".txt")
    with open(label_path, "w") as f:
        f.write("\n".join(lines))
    return split, name


def remove_from_dataset(abs_image_path, dirs):
    """Remove this image's copy and label file from the dataset, if present.

    Used for images whose shapes were all deleted since the last export, so a
    stale label file doesn't keep training the model on removed objects.
    Returns True if something was removed.
    """
    split, name = find_dataset_copy(abs_image_path, dirs)
    if split is None:
        return False
    images_dir, labels_dir = dirs[split]
    os.remove(os.path.join(images_dir, name))
    label_path = os.path.join(labels_dir, os.path.splitext(name)[0] + ".txt")
    if os.path.exists(label_path):
        os.remove(label_path)
    return True
