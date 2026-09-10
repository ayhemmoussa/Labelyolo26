import math
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from labelme.auto_label import auto_label_image
from labelme.yolo_export_common import dataset_dirs
from labelme.yolo_export_common import export_image
from labelme.yolo_export_common import remove_from_dataset
from labelme.yolo_export_common import rotation_corners
from labelme.yolo_export_common import shape_outline
from labelme.yolo_export_obb import _shape_to_obb_line

# ── OBB auto-label ────────────────────────────────────────────────────────────


def _obb_corners(cx, cy, w, h, angle_deg):
    """Corners in the same order as ultralytics' xywhr2xyxyxyxy."""
    a = math.radians(angle_deg)
    v1 = (w / 2 * math.cos(a), w / 2 * math.sin(a))
    v2 = (-h / 2 * math.sin(a), h / 2 * math.cos(a))
    return [
        (cx + v1[0] + v2[0], cy + v1[1] + v2[1]),
        (cx + v1[0] - v2[0], cy + v1[1] - v2[1]),
        (cx - v1[0] - v2[0], cy - v1[1] - v2[1]),
        (cx - v1[0] + v2[0], cy - v1[1] + v2[1]),
    ]


def _polygon_area(points):
    n = len(points)
    twice_area = sum(
        points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
        for i in range(n)
    )
    return abs(twice_area) / 2


class _FakeObbModel:
    """Stands in for an ultralytics YOLO OBB model with one detection."""

    names = {0: "seat"}

    def __init__(self, corners):
        self._corners = corners

    def __call__(self, image, **kwargs):
        obb = SimpleNamespace(
            xyxyxyxy=np.array([self._corners], dtype=np.float32),
            cls=np.array([0.0]),
            conf=np.array([0.9]),
        )
        return [SimpleNamespace(masks=None, boxes=None, obb=obb, names=self.names)]


def _auto_label_one_obb(tmp_path, corners):
    image_path = tmp_path / "img.png"
    cv2.imwrite(str(image_path), np.zeros((200, 200, 3), np.uint8))
    shapes = auto_label_image(_FakeObbModel(corners), str(image_path))
    assert len(shapes) == 1
    return shapes[0]


def test_auto_label_obb_keeps_predicted_corners(tmp_path):
    corners = _obb_corners(100, 80, 60, 20, 30)

    shape = _auto_label_one_obb(tmp_path, corners)

    assert shape["shape_type"] == "rotation"
    assert len(shape["points"]) == 5  # 4 corners + rotation handle
    assert np.allclose(shape["points"][:4], corners, atol=1e-3)
    # Regression: the old bbox+angle conversion inflated a 60x20 box to ~2900 px²
    assert _polygon_area(rotation_corners(shape)) == pytest.approx(60 * 20, rel=1e-3)


def test_obb_export_of_auto_label_matches_prediction(tmp_path):
    corners = _obb_corners(100, 80, 60, 20, 30)
    shape = _auto_label_one_obb(tmp_path, corners)

    line = _shape_to_obb_line(shape, ["seat"], 200, 200)

    values = [float(v) for v in line.split()]
    assert values[0] == 0
    exported = [(values[i] * 200, values[i + 1] * 200) for i in range(1, 9, 2)]
    assert np.allclose(exported, corners, atol=1e-2)


def test_rotation_corners_old_two_point_format():
    shape = {"points": [[0, 0], [20, 10]], "rotation": 90}

    corners = rotation_corners(shape)

    assert _polygon_area(corners) == pytest.approx(200)
    assert np.allclose(np.mean(corners, axis=0), (10, 5))


# ── Shape conversion ──────────────────────────────────────────────────────────


def test_shape_outline_converts_area_shapes():
    rect = shape_outline({"shape_type": "rectangle", "points": [[10, 20], [30, 40]]})
    assert rect == [(10, 20), (30, 20), (30, 40), (10, 40)]

    # 2-point polygons are boxes saved by older code paths
    box = shape_outline({"shape_type": "polygon", "points": [[10, 20], [30, 40]]})
    assert box == rect

    circle = shape_outline({"shape_type": "circle", "points": [[50, 50], [60, 50]]})
    assert len(circle) >= 16
    assert all(math.hypot(x - 50, y - 50) == pytest.approx(10) for x, y in circle)

    rotation = shape_outline(
        {"shape_type": "rotation", "points": [[0, 0], [10, 0], [10, 5], [0, 5], [5, -12]]}
    )
    assert rotation == [(0, 0), (10, 0), (10, 5), (0, 5)]  # handle point dropped


@pytest.mark.parametrize(
    "shape",
    [
        {"shape_type": "line", "points": [[0, 0], [10, 10]]},
        {"shape_type": "point", "points": [[5, 5]]},
        {"shape_type": "polygon", "points": [[0, 0]]},
        {"shape_type": "circle", "points": [[5, 5], [5, 5]]},
    ],
)
def test_shape_outline_skips_shapes_without_area(shape):
    assert shape_outline(shape) is None


# ── Dataset placement ─────────────────────────────────────────────────────────


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def _read(path):
    with open(path) as f:
        return f.read()


@pytest.fixture
def dirs(tmp_path):
    d = dataset_dirs(str(tmp_path / "dataset"))
    for images_dir, labels_dir in d.values():
        os.makedirs(images_dir)
        os.makedirs(labels_dir)
    return d


def test_same_filename_from_two_projects_does_not_overwrite(tmp_path, dirs):
    img_a = _write(tmp_path / "project_a" / "img_001.jpg", b"image A")
    img_b = _write(tmp_path / "project_b" / "img_001.jpg", b"image B")

    assert export_image(img_a, ["0 a"], "train", dirs) == ("train", "img_001.jpg")
    assert export_image(img_b, ["1 b"], "train", dirs) == ("train", "img_001_1.jpg")

    images_train, labels_train = dirs["train"]
    assert _read(os.path.join(labels_train, "img_001.txt")) == "0 a"
    assert _read(os.path.join(labels_train, "img_001_1.txt")) == "1 b"
    with open(os.path.join(images_train, "img_001.jpg"), "rb") as f:
        assert f.read() == b"image A"


def test_reexport_updates_label_in_place_and_keeps_split(tmp_path, dirs):
    img = _write(tmp_path / "project" / "img.jpg", b"image")
    export_image(img, ["0 old"], "val", dirs)

    assert export_image(img, ["0 new"], "train", dirs) == ("val", "img.jpg")

    assert os.listdir(dirs["train"][0]) == []
    assert _read(os.path.join(dirs["val"][1], "img.txt")) == "0 new"


def test_remove_from_dataset_drops_stale_copy(tmp_path, dirs):
    img = _write(tmp_path / "project" / "img.jpg", b"image")
    other = _write(tmp_path / "other" / "img.jpg", b"different image")
    export_image(img, ["0 a"], "train", dirs)

    assert remove_from_dataset(other, dirs) is False  # same name, different image
    assert remove_from_dataset(img, dirs) is True
    assert os.listdir(dirs["train"][0]) == []
    assert os.listdir(dirs["train"][1]) == []
