import copy
import math

import numpy as np
import skimage.measure
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui

import labelme.utils

# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.


class Shape(object):
    # Render handles as squares
    P_SQUARE = 0

    # Render handles as circles
    P_ROUND = 1

    # Flag for the handles we would move if dragging
    MOVE_VERTEX = 0

    # Flag for all other handles on the current shape
    NEAR_VERTEX = 1

    PEN_WIDTH = 2

    # The following class variables influence the drawing of all shape objects.
    line_color = None
    fill_color = None
    select_line_color = None
    select_fill_color = None
    vertex_fill_color = None
    hvertex_fill_color = None
    point_type = P_ROUND
    point_size = 8
    scale = 1.0

    def __init__(
        self,
        label=None,
        line_color=None,
        shape_type=None,
        flags=None,
        group_id=None,
        description=None,
        mask=None,
    ):
        self.label = label
        self.group_id = group_id
        self.points = []
        self.point_labels = []
        self.shape_type = shape_type
        self._shape_raw = None
        self._points_raw = []
        self._shape_type_raw = None
        self.fill = False
        self.selected = False
        self.flags = flags
        self.description = description
        self.other_data = {}
        self.mask = mask
        self.confidence = None  # float 0-1 from auto-labeling, None for manual

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (2.5, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_ROUND),  # round for consistency
        }

        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

    def _scale_point(self, point: QtCore.QPointF) -> QtCore.QPointF:
        return QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)

    def setShapeRefined(self, shape_type, points, point_labels, mask=None):
        self._shape_raw = (self.shape_type, self.points, self.point_labels)
        self.shape_type = shape_type
        self.points = points
        self.point_labels = point_labels
        self.mask = mask

    def restoreShapeRaw(self):
        if self._shape_raw is None:
            return
        self.shape_type, self.points, self.point_labels = self._shape_raw
        self._shape_raw = None

    @property
    def shape_type(self):
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        if value is None:
            value = "polygon"
        if value not in [
            "polygon",
            "rectangle",
            "point",
            "line",
            "circle",
            "linestrip",
            "points",
            "mask",
            "rotation",
        ]:
            raise ValueError("Unexpected shape_type: {}".format(value))
        self._shape_type = value

    def close(self):
        self._closed = True

    def addPoint(self, point, label=1):
        if self.points and point == self.points[0]:
            self.close()
        else:
            self.points.append(point)
            self.point_labels.append(label)

    def canAddPoint(self):
        return self.shape_type in ["polygon", "linestrip"]

    def popPoint(self):
        if self.points:
            if self.point_labels:
                self.point_labels.pop()
            return self.points.pop()
        return None

    def insertPoint(self, i, point, label=1):
        self.points.insert(i, point)
        self.point_labels.insert(i, label)

    def removePoint(self, i):
        if not self.canAddPoint():
            logger.warning(
                "Cannot remove point from: shape_type=%r",
                self.shape_type,
            )
            return

        if self.shape_type == "polygon" and len(self.points) <= 3:
            logger.warning(
                "Cannot remove point from: shape_type=%r, len(points)=%d",
                self.shape_type,
                len(self.points),
            )
            return

        if self.shape_type == "linestrip" and len(self.points) <= 2:
            logger.warning(
                "Cannot remove point from: shape_type=%r, len(points)=%d",
                self.shape_type,
                len(self.points),
            )
            return

        self.points.pop(i)
        self.point_labels.pop(i)

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    # ── Rotation-rectangle helpers ────────────────────────────────────────────

    def _rotation_corners(self):
        """Return the 4 corners of the rotated bounding rectangle in image space.

        New format (len >= 4): corners are stored directly as points[0..3].
        Old format (len < 4):  corners are computed from axis-aligned bbox + angle.
        """
        if len(self.points) >= 4:
            return list(self.points[0:4])
        # Backward-compat: compute from axis-aligned bbox corners + rotation angle
        if len(self.points) < 2:
            return []
        p0, p1 = self.points[0], self.points[1]
        cx = (p0.x() + p1.x()) / 2
        cy = (p0.y() + p1.y()) / 2
        hw = abs(p1.x() - p0.x()) / 2
        hh = abs(p1.y() - p0.y()) / 2
        angle = math.radians(self.other_data.get("rotation", 0.0))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        corners = []
        for lx, ly in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            corners.append(QtCore.QPointF(
                cx + lx * cos_a - ly * sin_a,
                cy + lx * sin_a + ly * cos_a,
            ))
        return corners

    def _rotation_top_center(self):
        """Return the rotated top-center point (midpoint of the top edge)."""
        corners = self._rotation_corners()
        if len(corners) < 2:
            return QtCore.QPointF(0, 0)
        return QtCore.QPointF(
            (corners[0].x() + corners[1].x()) / 2,
            (corners[0].y() + corners[1].y()) / 2,
        )

    def _rotation_handle_pos(self):
        """Return the position of the rotation handle (12 px outward from top-center)."""
        if len(self.points) >= 4:
            # New format: derive from stored corners
            corners = self.points[0:4]
            tx = (corners[0].x() + corners[1].x()) / 2
            ty = (corners[0].y() + corners[1].y()) / 2
            cx = sum(c.x() for c in corners) / 4
            cy = sum(c.y() for c in corners) / 4
            ddx, ddy = tx - cx, ty - cy
            dist = math.sqrt(ddx * ddx + ddy * ddy) or 1.0
            nx, ny = ddx / dist, ddy / dist
            return QtCore.QPointF(tx + nx * 12.0, ty + ny * 12.0)
        # Backward-compat: compute from axis-aligned bbox + angle
        if len(self.points) < 2:
            return QtCore.QPointF(0, 0)
        p0, p1 = self.points[0], self.points[1]
        cx = (p0.x() + p1.x()) / 2
        cy = (p0.y() + p1.y()) / 2
        hh = abs(p1.y() - p0.y()) / 2
        angle = math.radians(self.other_data.get("rotation", 0.0))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        lx, ly = 0.0, -(hh + 12.0)
        return QtCore.QPointF(
            cx + lx * cos_a - ly * sin_a,
            cy + lx * sin_a + ly * cos_a,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def paint(self, painter):
        if self.mask is None and not self.points:
            return

        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        color = self.select_line_color if self.selected else self.line_color
        pen = QtGui.QPen(color)
        pen.setWidthF(1.5)  # sub-pixel width for crisp antialiased lines
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        painter.setPen(pen)

        if self.mask is not None:
            image_to_draw = np.zeros(self.mask.shape + (4,), dtype=np.uint8)
            fill_color = (
                self.select_fill_color.getRgb()  # type: ignore[attr-defined]
                if self.selected
                else self.fill_color.getRgb()  # type: ignore[attr-defined]
            )
            image_to_draw[self.mask] = fill_color
            qimage = QtGui.QImage.fromData(labelme.utils.img_arr_to_data(image_to_draw))
            qimage = qimage.scaled(
                qimage.size() * self.scale,
                QtCore.Qt.IgnoreAspectRatio,  # type: ignore[attr-defined]
                QtCore.Qt.SmoothTransformation,  # type: ignore[attr-defined]
            )

            painter.drawImage(self._scale_point(point=self.points[0]), qimage)

            line_path = QtGui.QPainterPath()
            contours = skimage.measure.find_contours(np.pad(self.mask, pad_width=1))
            for contour in contours:
                contour += [self.points[0].y(), self.points[0].x()]
                line_path.moveTo(
                    self._scale_point(QtCore.QPointF(contour[0, 1], contour[0, 0]))
                )
                for point in contour[1:]:
                    line_path.lineTo(
                        self._scale_point(QtCore.QPointF(point[1], point[0]))
                    )
            painter.drawPath(line_path)

        if self.points:
            line_path = QtGui.QPainterPath()
            vrtx_path = QtGui.QPainterPath()
            negative_vrtx_path = QtGui.QPainterPath()

            if self.shape_type == "rectangle":
                _r = 3  # rounded corner radius
                if len(self.points) == 2:
                    rectangle = QtCore.QRectF(
                        self._scale_point(self.points[0]),
                        self._scale_point(self.points[1]),
                    )
                    line_path.addRoundedRect(rectangle, _r, _r)
                    for i in range(len(self.points)):
                        self.drawVertex(vrtx_path, i)
                elif len(self.points) == 4:
                    xs = [p.x() for p in self.points]
                    ys = [p.y() for p in self.points]
                    top_left = self._scale_point(QtCore.QPointF(min(xs), min(ys)))
                    bottom_right = self._scale_point(QtCore.QPointF(max(xs), max(ys)))
                    rect = QtCore.QRectF(top_left, bottom_right)
                    line_path.addRoundedRect(rect, _r, _r)
                    for i in range(len(self.points)):
                        self.drawVertex(vrtx_path, i)
                elif len(self.points) > 0:
                    logger.debug(f"Rectangle shape skipped in paint: {len(self.points)} point(s)")
            elif self.shape_type == "rotation":
                corners = self._rotation_corners()
                if corners:
                    line_path.moveTo(self._scale_point(corners[0]))
                    for c in corners[1:]:
                        line_path.lineTo(self._scale_point(c))
                    line_path.closeSubpath()
                    # Draw small square handles at the 4 visible corners
                    n_corners = min(4, len(self.points))
                    for i in range(n_corners):
                        self.drawVertex(vrtx_path, i)
                    # Stem line and rotation handle only on finalized shapes (5-point format).
                    # During drawing preview the shape has 2 points; skip stem to avoid a
                    # dangling line with no handle circle.
                    handle_idx = 4 if len(self.points) >= 5 else None
                    if handle_idx is not None:
                        top_center = self._rotation_top_center()
                        handle_pos = self._rotation_handle_pos()
                        painter.drawLine(
                            self._scale_point(top_center),
                            self._scale_point(handle_pos),
                        )
                    if handle_idx is not None:
                        self._vertex_fill_color = (
                            self.hvertex_fill_color
                            if self._highlightIndex is not None
                            else self.vertex_fill_color
                        )
                        handle_pt = self._scale_point(self.points[handle_idx])
                        handle_r = self.point_size * 1.8
                        vrtx_path.addEllipse(handle_pt, handle_r, handle_r)
            elif self.shape_type == "mask":
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = QtCore.QRectF(
                        self._scale_point(self.points[0]),
                        self._scale_point(self.points[1]),
                    )
                    line_path.addRect(rectangle)

            elif self.shape_type == "circle":
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    raidus = labelme.utils.distance(
                        self._scale_point(self.points[0] - self.points[1])
                    )
                    line_path.addEllipse(
                        self._scale_point(self.points[0]), raidus, raidus
                    )
                for i in range(len(self.points)):
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "linestrip":
                line_path.moveTo(self._scale_point(self.points[0]))
                for i, p in enumerate(self.points):
                    line_path.lineTo(self._scale_point(p))
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "points":
                assert len(self.points) == len(self.point_labels)
                for i, point_label in enumerate(self.point_labels):
                    if point_label == 1:
                        self.drawVertex(vrtx_path, i)
                    else:
                        self.drawVertex(negative_vrtx_path, i)
            else:
                line_path.moveTo(self._scale_point(self.points[0]))
                # Uncommenting the following line will draw 2 paths
                # for the 1st vertex, and make it non-filled, which
                # may be desirable.
                # self.drawVertex(vrtx_path, 0)

                for i, p in enumerate(self.points):
                    line_path.lineTo(self._scale_point(p))
                    self.drawVertex(vrtx_path, i)
                if self.isClosed():
                    line_path.lineTo(self._scale_point(self.points[0]))

            painter.drawPath(line_path)
            if vrtx_path.length() > 0:
                painter.fillPath(vrtx_path, self._vertex_fill_color)  # type: ignore[has-type]
                vrtx_pen = QtGui.QPen(color)
                vrtx_pen.setWidthF(1.0)
                vrtx_pen.setCapStyle(QtCore.Qt.RoundCap)
                painter.setPen(vrtx_pen)
                painter.drawPath(vrtx_path)
                painter.setPen(pen)
            if self.fill and self.mask is None:
                color = self.select_fill_color if self.selected else self.fill_color
                painter.fillPath(line_path, color)

            pen.setColor(QtGui.QColor(255, 0, 0, 255))
            painter.setPen(pen)
            painter.drawPath(negative_vrtx_path)
            painter.fillPath(negative_vrtx_path, QtGui.QColor(255, 0, 0, 255))

        # Draw confidence label if available
        if self.confidence is not None and self.points:
            conf_text = f"{self.confidence:.0%}"
            pt = self._scale_point(self.points[0])
            conf_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 200))
            painter.setPen(conf_pen)
            font = QtGui.QFont()
            font.setPointSize(7)
            font.setBold(True)
            painter.setFont(font)
            # Small background rect
            fm = QtGui.QFontMetricsF(font)
            text_w = fm.width(conf_text)
            text_h = fm.height()
            bg_rect = QtCore.QRectF(pt.x() + 3, pt.y() - text_h - 2, text_w + 6, text_h + 2)
            painter.fillRect(bg_rect, QtGui.QColor(0, 0, 0, 140))
            painter.drawText(bg_rect, QtCore.Qt.AlignCenter, conf_text)

    def drawVertex(self, path, i):
        d = self.point_size
        shape = self.point_type
        point = self._scale_point(self.points[i])
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size  # type: ignore[assignment]
        if self._highlightIndex is not None:
            self._vertex_fill_color = self.hvertex_fill_color
        else:
            self._vertex_fill_color = self.vertex_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def nearestVertex(self, point, epsilon):
        min_distance = float("inf")
        min_i = None
        point = QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)
        for i, p in enumerate(self.points):
            p = QtCore.QPointF(p.x() * self.scale, p.y() * self.scale)
            dist = labelme.utils.distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i
        return min_i

    def nearestEdge(self, point, epsilon):
        min_distance = float("inf")
        post_i = None
        point = QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)
        for i in range(len(self.points)):
            start = self.points[i - 1]
            end = self.points[i]
            start = QtCore.QPointF(start.x() * self.scale, start.y() * self.scale)
            end = QtCore.QPointF(end.x() * self.scale, end.y() * self.scale)
            line = [start, end]
            dist = labelme.utils.distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

    def containsPoint(self, point):
        if self.mask is not None:
            y = np.clip(
                int(round(point.y() - self.points[0].y())),
                0,
                self.mask.shape[0] - 1,
            )
            x = np.clip(
                int(round(point.x() - self.points[0].x())),
                0,
                self.mask.shape[1] - 1,
            )
            return self.mask[y, x]
        return self.makePath().contains(point)

    def makePath(self):
        if self.shape_type == "rotation":
            path = QtGui.QPainterPath()
            corners = self._rotation_corners()
            if corners:
                path.moveTo(corners[0])
                for c in corners[1:]:
                    path.lineTo(c)
                path.closeSubpath()
            return path
        elif self.shape_type == "rectangle":
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                path.addRect(QtCore.QRectF(self.points[0], self.points[1]))
            elif len(self.points) == 4:
                xs = [p.x() for p in self.points]
                ys = [p.y() for p in self.points]
                top_left = QtCore.QPointF(min(xs), min(ys))
                bottom_right = QtCore.QPointF(max(xs), max(ys))
                path.addRect(QtCore.QRectF(top_left, bottom_right))
        elif self.shape_type == "mask":
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                path.addRect(QtCore.QRectF(self.points[0], self.points[1]))
        elif self.shape_type == "circle":
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                raidus = labelme.utils.distance(self.points[0] - self.points[1])
                path.addEllipse(self.points[0], raidus, raidus)
        else:
            path = QtGui.QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
        return path

    def boundingRect(self):
        return self.makePath().boundingRect()

    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]
        # For rotation shapes the handle is derived — recompute it after translation.
        if self.shape_type == "rotation":
            if len(self.points) >= 5:
                self.points[4] = self._rotation_handle_pos()
            elif len(self.points) >= 3:
                self.points[2] = self._rotation_handle_pos()

    def moveVertexBy(self, i, offset):
        if self.shape_type == "rotation" and len(self.points) >= 5:
            if i == 4:
                # Rotation handle dragged — rotate all 4 corners around the center
                new_handle = self.points[4] + offset
                corners = self.points[0:4]
                cx = sum(c.x() for c in corners) / 4
                cy = sum(c.y() for c in corners) / 4
                dx = new_handle.x() - cx
                dy = new_handle.y() - cy
                new_angle = math.degrees(math.atan2(dx, -dy))
                old_angle = self.other_data.get("rotation", 0.0)
                delta_rad = math.radians(new_angle - old_angle)
                cos_d = math.cos(delta_rad)
                sin_d = math.sin(delta_rad)
                new_corners = []
                for c in corners:
                    lx = c.x() - cx
                    ly = c.y() - cy
                    new_corners.append(QtCore.QPointF(
                        cx + lx * cos_d - ly * sin_d,
                        cy + lx * sin_d + ly * cos_d,
                    ))
                self.points[0:4] = new_corners
                self.other_data["rotation"] = new_angle
                self.points[4] = self._rotation_handle_pos()
            elif 0 <= i <= 3:
                # Corner dragged — resize, keeping the opposite corner fixed
                corners = list(self.points[0:4])
                new_corner = corners[i] + offset
                opp = (i + 2) % 4
                opp_corner = corners[opp]
                # New center = midpoint of (dragged corner, opposite corner)
                new_cx = (new_corner.x() + opp_corner.x()) / 2
                new_cy = (new_corner.y() + opp_corner.y()) / 2
                angle_rad = math.radians(self.other_data.get("rotation", 0.0))
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)
                # Project new_corner (relative to new center) onto local axes
                dx = new_corner.x() - new_cx
                dy = new_corner.y() - new_cy
                local_x = dx * cos_a + dy * sin_a
                local_y = -dx * sin_a + dy * cos_a
                new_hw = max(5.0, abs(local_x))
                new_hh = max(5.0, abs(local_y))
                # Rebuild all 4 corners from new center + half-dims + angle
                new_corners = []
                for lx, ly in [(-new_hw, -new_hh), (new_hw, -new_hh),
                                (new_hw, new_hh), (-new_hw, new_hh)]:
                    new_corners.append(QtCore.QPointF(
                        new_cx + lx * cos_a - ly * sin_a,
                        new_cy + lx * sin_a + ly * cos_a,
                    ))
                self.points[0:4] = new_corners
                self.points[4] = self._rotation_handle_pos()
        elif self.shape_type == "rotation" and len(self.points) == 3:
            # Backward-compat old 3-point format
            if i == 2:
                new_handle = self.points[2] + offset
                p0, p1 = self.points[0], self.points[1]
                cx = (p0.x() + p1.x()) / 2
                cy = (p0.y() + p1.y()) / 2
                dx = new_handle.x() - cx
                dy = new_handle.y() - cy
                new_angle = math.degrees(math.atan2(dx, -dy))
                self.other_data["rotation"] = new_angle
                self.points[2] = self._rotation_handle_pos()
            else:
                self.points[i] = self.points[i] + offset
                self.points[2] = self._rotation_handle_pos()
        else:
            self.points[i] = self.points[i] + offset

    def highlightVertex(self, i, action):
        """Highlight a vertex appropriately based on the current action

        Args:
            i (int): The vertex index
            action (int): The action
            (see Shape.NEAR_VERTEX and Shape.MOVE_VERTEX)
        """
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        """Clear the highlighted point"""
        self._highlightIndex = None

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value
