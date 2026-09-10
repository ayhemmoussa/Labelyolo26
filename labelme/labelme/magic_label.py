"""
Magic Label - Auto-labeling using Grounding DINO + SAM2.

Pipeline:
  1. Grounding DINO (text-prompted): detects bounding boxes from free-text labels.
  2. SAM2 (optional): refines each box into a precise segmentation polygon.

Dependencies (installed separately):
  pip install transformers          # for Grounding DINO
  pip install sam2                  # for SAM2 mask generation (optional)
"""

import os
import sys
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

# Suppress noisy HuggingFace / transformers startup warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*use_fast.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*text_labels.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*labels.*will return integer.*", category=FutureWarning)


def _models_dir() -> str:
    """Return the bundled models directory regardless of frozen/source context."""
    if getattr(sys, "frozen", False):
        # PyInstaller .exe — models are next to the executable bundle
        return os.path.join(sys._MEIPASS, "labelme", "models")
    # Running from source — models live beside this file
    return os.path.join(os.path.dirname(__file__), "models")


# Bundled model paths
_LOCAL_GDINO_DIR = os.path.join(_models_dir(), "grounding_dino_tiny")
_LOCAL_SAM2_CKPT = os.path.join(_models_dir(), "sam2", "sam2_hiera_tiny.pt")

# ---------------------------------------------------------------------------
# Module-level model cache (loaded once per session)
# ---------------------------------------------------------------------------
_gdino_model = None
_gdino_processor = None
_sam2_predictor = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_grounding_dino(progress_callback=None):
    """Load (and cache) the Grounding DINO model — local copy first, HF fallback."""
    global _gdino_model, _gdino_processor
    if _gdino_model is not None:
        return _gdino_model, _gdino_processor

    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError:
        raise ImportError(
            "The 'transformers' package is required for Magic Label.\n\n"
            "Install it with:\n"
            "    pip install transformers\n\n"
            "Then restart the application."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.isdir(_LOCAL_GDINO_DIR):
        model_source = _LOCAL_GDINO_DIR
        if progress_callback:
            progress_callback(f"Loading Grounding DINO from local cache (device={device})…")
    else:
        model_source = "IDEA-Research/grounding-dino-tiny"
        if progress_callback:
            progress_callback("Downloading Grounding DINO model (first run only)…")

    _gdino_processor = AutoProcessor.from_pretrained(model_source)
    _gdino_model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(model_source).to(device).eval()
    )
    return _gdino_model, _gdino_processor


def _load_sam2(progress_callback=None):
    """Load (and cache) SAM2 — local checkpoint first, HF fallback."""
    global _sam2_predictor
    if _sam2_predictor is not None:
        return _sam2_predictor

    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import build_sam2
    except ImportError:
        raise ImportError(
            "The 'sam2' package is required for SAM2 mask generation.\n\n"
            "Install it with:\n"
            "    pip install sam2\n\n"
            "Or uncheck 'Use SAM2 masks' to use bounding boxes instead."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.isfile(_LOCAL_SAM2_CKPT):
        if progress_callback:
            progress_callback(f"Loading SAM2 from local checkpoint (device={device})…")
        sam_model = build_sam2(
            config_file="configs/sam2/sam2_hiera_t.yaml",
            ckpt_path=_LOCAL_SAM2_CKPT,
            device=device,
        )
        _sam2_predictor = SAM2ImagePredictor(sam_model)
    else:
        if progress_callback:
            progress_callback(f"Downloading SAM2 model (first run only, device={device})…")
        _sam2_predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-tiny", device=device
        )

    return _sam2_predictor


def _box_to_points(box, img_w, img_h):
    """Convert xyxy box to [top-left, bottom-right] for a labelme rectangle."""
    xmin = float(np.clip(box[0], 0, img_w - 1))
    ymin = float(np.clip(box[1], 0, img_h - 1))
    xmax = float(np.clip(box[2], 0, img_w - 1))
    ymax = float(np.clip(box[3], 0, img_h - 1))
    return [(xmin, ymin), (xmax, ymax)]


def _mask_to_polygon(mask_bool, img_w, img_h):
    """
    Convert a boolean mask (H x W) to a polygon as a list of (x, y) tuples.
    Returns None if no valid contour is found.
    """
    mask_u8 = mask_bool.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    return [
        (float(np.clip(x, 0, img_w - 1)), float(np.clip(y, 0, img_h - 1)))
        for [[x, y]] in contour
    ]


def _make_shape(label, points, shape_type="polygon"):
    """Build a shape dict in the format expected by labelme."""
    return {
        "label": label,
        "points": points,
        "shape_type": shape_type,
        "flags": {},
        "group_id": None,
        "description": "",
        "other_data": {},
        "mask": None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class StopRequested(Exception):
    """Raised internally when the caller requests cancellation."""


def magic_label_image(
    image_path,
    text_prompt,
    box_threshold=0.30,
    text_threshold=0.25,
    use_sam=True,
    progress_callback=None,
    stop_callback=None,
):
    """
    Auto-label an image using Grounding DINO + optional SAM2.

    Args:
        image_path      : Path to the image file.
        text_prompt     : Comma-separated class names,
                          e.g. "car seat, buckle, foam headrest".
        box_threshold   : Grounding DINO box confidence threshold (0–1).
        text_threshold  : Grounding DINO text-similarity threshold (0–1).
        use_sam         : If True, use SAM2 for precise segmentation masks.
                          Falls back to bounding boxes if sam2 is not installed.
        progress_callback: Optional callable(str) for status messages.
        stop_callback   : Optional callable() → bool.  Return True to cancel.

    Returns:
        List of shape dicts compatible with labelme's shape format.
        Returns [] if cancelled.

    Raises:
        FileNotFoundError : If the image does not exist.
        ImportError       : If 'transformers' is not installed.
    """
    def _check_stop():
        if stop_callback and stop_callback():
            raise StopRequested()

    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # --- Load image ---
        image_pil = Image.open(image_path).convert("RGB")
        img_w, img_h = image_pil.size

        # --- Build Grounding DINO text prompt ---
        # The model expects period-separated labels ending with a period.
        labels = [lbl.strip() for lbl in text_prompt.replace(".", ",").split(",") if lbl.strip()]
        if not labels:
            return []
        gdino_text = ". ".join(labels) + "."

        # --- Step 1: Grounding DINO detection ---
        if progress_callback:
            progress_callback("Running Grounding DINO detection…")

        gdino_model, gdino_processor = _load_grounding_dino(progress_callback)
        _check_stop()
        device = next(gdino_model.parameters()).device

        inputs = gdino_processor(images=image_pil, text=gdino_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = gdino_model(**inputs)

        results = gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,       # renamed in transformers 5.x (was box_threshold)
            text_threshold=text_threshold,
            target_sizes=[(img_h, img_w)],
        )[0]

        boxes_t = results.get("boxes", torch.zeros(0, 4))
        # transformers ≥ 4.51 renames "labels" (int ids) → "text_labels" (strings)
        raw_labels = results.get("text_labels") or results.get("labels", [])
        boxes = boxes_t.cpu().numpy()  # [N, 4] xyxy in image coordinates

        # GDINO tokenizes multi-word labels and may return only the first token
        # (e.g. "car" instead of "car seat"). Remap each detected label back to
        # the closest user-specified label using substring/token overlap.
        def _remap_label(detected, user_labels):
            d = detected.lower().strip()
            # exact match
            for lbl in user_labels:
                if lbl.lower() == d:
                    return lbl
            # detected token is a word inside a user label  ("car" → "car seat")
            for lbl in user_labels:
                if d in lbl.lower().split():
                    return lbl
            # user label starts with the detected string
            for lbl in user_labels:
                if lbl.lower().startswith(d):
                    return lbl
            # any overlap
            for lbl in user_labels:
                if d in lbl.lower() or lbl.lower() in d:
                    return lbl
            return detected  # give up, keep as-is

        det_labels = [_remap_label(lbl, labels) for lbl in raw_labels]

        _check_stop()

        if len(boxes) == 0:
            return []

        shapes = []

        # --- Step 2: SAM2 segmentation (optional) ---
        if use_sam:
            try:
                if progress_callback:
                    progress_callback("Loading SAM2 and generating masks…")

                sam2_predictor = _load_sam2(progress_callback)
                _check_stop()
                image_np = np.array(image_pil)

                with torch.inference_mode():
                    sam2_predictor.set_image(image_np)

                    for i, (box, label) in enumerate(zip(boxes, det_labels)):
                        _check_stop()
                        if progress_callback:
                            progress_callback(f"SAM2 masking object {i + 1}/{len(boxes)}: {label}…")
                        try:
                            masks, _, _ = sam2_predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=box,
                                multimask_output=False,
                            )
                            mask_bool = masks[0] > 0  # (H, W) boolean
                            points = _mask_to_polygon(mask_bool, img_w, img_h)
                            if points:
                                shapes.append(_make_shape(label, points, "polygon"))
                                continue
                        except Exception as e:
                            print(f"[MAGIC LABEL] SAM2 failed for '{label}': {e}")
                        # Per-detection fallback: use a rectangle
                        shapes.append(_make_shape(label, _box_to_points(box, img_w, img_h), "rectangle"))

                return shapes  # SAM2 path succeeded (possibly with per-box fallbacks)

            except StopRequested:
                raise
            except ImportError as e:
                print(f"[MAGIC LABEL] sam2 not installed — using bounding boxes.\n  {e}")
            except Exception as e:
                print(f"[MAGIC LABEL] SAM2 error — using bounding boxes.\n  {e}")

        # --- Rectangle-only fallback ---
        for box, label in zip(boxes, det_labels):
            _check_stop()
            shapes.append(_make_shape(label, _box_to_points(box, img_w, img_h), "rectangle"))

        return shapes

    except StopRequested:
        return []


def clear_model_cache():
    """Release cached models to free GPU/CPU memory."""
    global _gdino_model, _gdino_processor, _sam2_predictor
    _gdino_model = None
    _gdino_processor = None
    _sam2_predictor = None
