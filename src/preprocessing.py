"""
Preprocessing pipeline - identical to the training notebook:

    1. Face detection with OpenCV Haar Cascade
       (frontalface_default, fallback to frontalface_alt2)
    2. Histogram equalisation on the grayscale image before detection
    3. Largest detected face is cropped with a 30 % margin
    4. BGR -> RGB conversion
    5. Resize to 224 x 224
    6. Convert to float32 and normalise
         - Custom CNN        : pixels scaled to [-1, 1]
         - Transfer learning : ImageNet mean / std

If no face is detected the whole frame is square-cropped and resized so the
live demo keeps running (the result is flagged with ``face_detected=False``).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

IMG_SIZE = (224, 224)
MARGIN = 0.30

# Normalisation presets -------------------------------------------------------
NORMALIZATION_STATS = {
    # pixels in [0,1] -> (x - 0.5) / 0.5  ==  range [-1, 1]
    "minus1_1": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ImageNet statistics (torchvision pretrained backbones)
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}

# Display colours (BGR) per predicted class.
LABEL_COLORS_BGR = {
    "Real": (46, 204, 113),    # green
    "Print": (231, 76, 60),    # red
    "Replay": (243, 156, 18),  # orange
}

_CASCADES: tuple[cv2.CascadeClassifier, cv2.CascadeClassifier] | None = None


# --------------------------------------------------------------------------- #
# Haar cascades (loaded once, reused across reruns)
# --------------------------------------------------------------------------- #
def _get_cascades():
    global _CASCADES
    if _CASCADES is None:
        default = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        alt2 = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )
        _CASCADES = (default, alt2)
    return _CASCADES


def detect_face(image_bgr: np.ndarray):
    """Return the largest face box ``(x1, y1, x2, y2)`` or ``None``."""
    if image_bgr is None or image_bgr.size == 0:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    default, alt2 = _get_cascades()

    faces = default.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
    )

    if len(faces) == 0:
        faces = alt2.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
        )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

    margin_x = int(w * MARGIN)
    margin_y = int(h * MARGIN)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image_bgr.shape[1], x + w + margin_x)
    y2 = min(image_bgr.shape[0], y + h + margin_y)

    return int(x1), int(y1), int(x2), int(y2)


def _center_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


def detect_and_crop(image_bgr: np.ndarray, size=IMG_SIZE, margin=MARGIN):
    """Run the training-time pipeline on a BGR frame.

    Returns ``(rgb_uint8, box_or_None, face_detected)`` where ``box`` is in the
    coordinates of the original image (for drawing the bounding box).
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to detect_and_crop().")

    box = detect_face(image_bgr)
    face_detected = box is not None

    if face_detected:
        x1, y1, x2, y2 = box
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image_bgr
            box = None
            face_detected = False
    else:
        # Fallback: keep the demo alive when detection fails.
        crop = _center_square(image_bgr)
        box = None

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return rgb, box, face_detected


def to_tensor(rgb: np.ndarray, normalization: str = "minus1_1") -> torch.Tensor:
    """H x W x 3 uint8 RGB -> normalised 3 x 224 x 224 float tensor."""
    if normalization not in NORMALIZATION_STATS:
        raise ValueError(
            f"Unknown normalization '{normalization}'. "
            f"Available: {', '.join(NORMALIZATION_STATS)}"
        )

    mean, std = NORMALIZATION_STATS[normalization]
    x = rgb.astype(np.float32) / 255.0

    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    x = (x - mean_arr) / std_arr

    # HWC -> CHW, contiguous so torch does not need to copy it again.
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


# --------------------------------------------------------------------------- #
# Overlay drawing for the live demo
# --------------------------------------------------------------------------- #
def _put_tag(image, text, org, color, scale=0.65, thickness=2):
    """Draw text with a filled background rectangle."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x, y = org
    cv2.rectangle(
        image,
        (x - 4, y - th - 6),
        (x + tw + 4, y + baseline + 4),
        color,
        -1,
    )
    cv2.putText(
        image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
        (255, 255, 255), thickness, cv2.LINE_AA,
    )


def annotate_frame(
    frame_bgr: np.ndarray,
    box,
    label: str,
    confidence: float,
    status: str | None = None,
    face_detected: bool = True,
) -> np.ndarray:
    """Return a copy of the frame with bounding box, label and status text."""
    vis = frame_bgr.copy()
    color = LABEL_COLORS_BGR.get(label, (200, 200, 200))

    if face_detected and box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        _put_tag(vis, f"{label}: {confidence:.1%}", (max(4, x1), max(24, y1 - 8)), color)
    else:
        _put_tag(vis, f"{label}: {confidence:.1%}  (no face detected)", (8, 26), color)

    if status:
        status_color = {
            "GRANTED": (46, 204, 113),
            "DENIED": (231, 76, 60),
            "UNCERTAIN": (243, 156, 18),
        }.get(status.split(" ")[0].upper(), (90, 90, 90))
        _put_tag(vis, status, (8, vis.shape[0] - 14), status_color, scale=0.75)

    return vis
