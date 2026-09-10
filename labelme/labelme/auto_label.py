import math
import cv2
import numpy as np
import os


def to_numpy(x):
    """Safely convert tensor/array/list to numpy array on CPU."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.array(x)
    # torch tensor
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.array(x)


def extract_mask_polygons(mask):
    """
    mask: either a Nx2 array of coords or a torch tensor.
    Returns list of (x, y) pairs as floats.
    """
    arr = to_numpy(mask)
    if arr is None:
        return []
    if arr.ndim == 1:
        return [(float(arr[0]), float(arr[1]))]
    return [(float(x), float(y)) for x, y in arr]


def _clip_point(x, y, width, height):
    """Clip a point to valid image coordinates."""
    return (
        float(np.clip(x, 0, max(width - 1, 0))),
        float(np.clip(y, 0, max(height - 1, 0))),
    )


def auto_label_image(yolo_model, image_path, class_list=None, img_size=640, conf_threshold=0.35):
    """Auto-label an image using an Ultralytics YOLO model.

    Args:
        yolo_model: Loaded ultralytics.YOLO model instance.
        image_path: Path to the image file.
        class_list: List of class names (optional; model's own names are used if available).
        img_size: Input image size for model inference (default: 640).
        conf_threshold: Confidence threshold for detections (default: 0.35).

    Returns:
        list: Detected shapes with labels and coordinates.

    Raises:
        ValueError: If model or parameters are invalid.
        FileNotFoundError: If image file does not exist or cannot be read.
        RuntimeError: If inference fails.
    """
    if yolo_model is None:
        raise ValueError("YOLO model is None. Please load a valid model first.")

    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image file not found or invalid path:\n{image_path}\n\n"
            "Please ensure the image file exists and the path is correct."
        )

    if conf_threshold < 0 or conf_threshold > 1:
        raise ValueError(
            f"Confidence threshold must be between 0 and 1, got: {conf_threshold}"
        )

    # Prefer names from the model object itself
    names = getattr(yolo_model, "names", None) or class_list

    # Load image
    try:
        im_bgr = cv2.imread(image_path)
        if im_bgr is None:
            raise FileNotFoundError(
                f"Failed to load image: {os.path.basename(image_path)}\n\n"
                "Possible causes:\n"
                "• File is corrupted or not a valid image format\n"
                "• File path contains unsupported characters\n"
                "• Insufficient permissions to read the file\n\n"
                f"Full path: {image_path}"
            )
        h0, w0 = im_bgr.shape[:2]
        if h0 == 0 or w0 == 0:
            raise ValueError(
                f"Invalid image dimensions: {w0}x{h0}\n"
                f"Image: {os.path.basename(image_path)}"
            )
    except cv2.error as e:
        raise RuntimeError(
            f"OpenCV error while loading image:\n{str(e)}\n\n"
            f"Image: {os.path.basename(image_path)}"
        ) from e

    # Run inference via the ultralytics high-level API.
    # The model handles its own preprocessing; results are already in original-image space.
    try:
        results_list = yolo_model(
            im_bgr,
            conf=conf_threshold,
            imgsz=img_size,
            verbose=False,
        )
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "cuda" in error_msg:
            raise RuntimeError(
                "GPU Out of Memory Error\n\n"
                "The model ran out of GPU memory during inference.\n\n"
                "Solutions:\n"
                "• Try using a smaller image size\n"
                "• Close other GPU-intensive applications\n"
                "• Restart the application to clear GPU cache\n\n"
                f"Technical details: {str(e)}"
            ) from e
        raise RuntimeError(
            f"Model inference failed:\n{str(e)}\n\n"
            "Possible causes:\n"
            "• Model file is corrupted\n"
            "• Model is incompatible with current image\n"
            "• Missing model dependencies"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error during model inference:\n{str(e)}\n\n"
            "Please check that the model is properly loaded and compatible."
        ) from e

    shapes = []

    for res in results_list:
        masks_obj = getattr(res, "masks", None)
        boxes_obj = getattr(res, "boxes", None)
        obb_obj   = getattr(res, "obb",   None)
        # Prefer per-result names, fall back to model-level names
        res_names = getattr(res, "names", None) or names

        if masks_obj is not None and hasattr(masks_obj, "xy"):
            # --- Segmentation result ---
            polygons = masks_obj.xy
            classes = getattr(boxes_obj, "cls", None) if boxes_obj is not None else None
            confs = getattr(boxes_obj, "conf", None) if boxes_obj is not None else None

            classes_np = to_numpy(classes) if classes is not None else None
            confs_np = to_numpy(confs) if confs is not None else None

            for i, poly in enumerate(polygons):
                conf_val = float(confs_np[i]) if confs_np is not None else 1.0
                cls_i = int(classes_np[i]) if classes_np is not None else 0

                if isinstance(res_names, dict):
                    label = res_names.get(cls_i, str(cls_i))
                else:
                    label = (
                        res_names[cls_i]
                        if res_names and cls_i < len(res_names)
                        else str(cls_i)
                    )

                points = extract_mask_polygons(poly)
                if len(points) < 3:
                    print(f"[WARNING] Skipping mask {i}: insufficient polygon points ({len(points)})")
                    continue

                # Clip polygon points to image bounds
                points = [_clip_point(x, y, w0, h0) for x, y in points]

                shapes.append({
                    "label": label,
                    "points": points,
                    "shape_type": "polygon",
                    "flags": {},
                    "group_id": None,
                    "description": "",
                    "other_data": {},
                    "mask": None,
                    "confidence": conf_val,
                })

        elif obb_obj is not None and hasattr(obb_obj, "xyxyxyxy"):
            # --- OBB (Oriented Bounding Box) result ---
            corners_tensor = obb_obj.xyxyxyxy   # shape (N, 4, 2)
            classes = getattr(obb_obj, "cls",  None)
            confs   = getattr(obb_obj, "conf", None)

            corners_np  = to_numpy(corners_tensor) if corners_tensor is not None else None
            classes_np  = to_numpy(classes)         if classes  is not None else None
            confs_np    = to_numpy(confs)            if confs    is not None else None

            if corners_np is not None:
                for i, corners in enumerate(corners_np):
                    conf_val = float(confs_np[i])   if confs_np   is not None else 1.0
                    cls_i    = int(classes_np[i])   if classes_np is not None else 0

                    if isinstance(res_names, dict):
                        label = res_names.get(cls_i, str(cls_i))
                    else:
                        label = (
                            res_names[cls_i]
                            if res_names and cls_i < len(res_names)
                            else str(cls_i)
                        )

                    # corners is shape (4, 2): four (x, y) pixel-space corners.
                    # Convert to a "rotation" shape: axis-aligned bbox + angle,
                    # which matches what the OBB canvas tool produces and exports.
                    raw = [
                        _clip_point(float(corners[j, 0]), float(corners[j, 1]), w0, h0)
                        for j in range(4)
                    ]
                    xs = [p[0] for p in raw]
                    ys = [p[1] for p in raw]
                    bx0, bx1 = min(xs), max(xs)
                    by0, by1 = min(ys), max(ys)
                    # Estimate rotation from the first edge direction
                    dx = raw[1][0] - raw[0][0]
                    dy = raw[1][1] - raw[0][1]
                    angle_deg = math.degrees(math.atan2(dy, dx))

                    shapes.append({
                        "label":       label,
                        "points":      [(bx0, by0), (bx1, by1)],
                        "shape_type":  "rotation",
                        "flags":       {},
                        "group_id":    None,
                        "description": "",
                        "other_data":  {"rotation": angle_deg},
                        "mask":        None,
                        "confidence":  conf_val,
                    })

        elif boxes_obj is not None:
            # --- Detection / bounding-box result ---
            boxes_xyxy = getattr(boxes_obj, "xyxy", None)
            classes = getattr(boxes_obj, "cls", None)
            confs = getattr(boxes_obj, "conf", None)

            boxes_np = to_numpy(boxes_xyxy) if boxes_xyxy is not None else None
            classes_np = to_numpy(classes) if classes is not None else None
            confs_np = to_numpy(confs) if confs is not None else None

            if boxes_np is not None:
                for i, b in enumerate(boxes_np):
                    conf_val = float(confs_np[i]) if confs_np is not None else 1.0
                    cls_i = int(classes_np[i]) if classes_np is not None else 0

                    if isinstance(res_names, dict):
                        label = res_names.get(cls_i, str(cls_i))
                    else:
                        label = (
                            res_names[cls_i]
                            if res_names and cls_i < len(res_names)
                            else str(cls_i)
                        )

                    xmin, ymin, xmax, ymax = [float(v) for v in b[:4]]
                    # Clip to image bounds
                    xmin, ymin = _clip_point(xmin, ymin, w0, h0)
                    xmax, ymax = _clip_point(xmax, ymax, w0, h0)

                    shapes.append({
                        "label": label,
                        "points": [(xmin, ymin), (xmax, ymax)],
                        "shape_type": "rectangle",
                        "flags": {},
                        "group_id": None,
                        "description": "",
                        "other_data": {},
                        "mask": None,
                        "confidence": conf_val,
                    })

    return shapes
