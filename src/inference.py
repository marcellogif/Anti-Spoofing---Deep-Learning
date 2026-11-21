"""
Checkpoint discovery/loading, single-image prediction and efficiency
benchmarking helpers for the Face Anti-Spoofing Streamlit app.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .models import (
    CLASS_NAMES,
    NORMALIZATION,
    build_model,
    count_parameters,
    model_size_mb,
)
from .preprocessing import detect_and_crop, to_tensor

DEFAULT_CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

# Substrings used to match checkpoint file names to architectures.
_FILE_PATTERNS = {
    "Custom CNN": ("customcnn", "baselinecnn", "cnnmodel"),
    "MobileNetV2": ("mobilenetv2", "mobilenet"),
    "ResNet18": ("resnet18",),
    "EfficientNetB0": ("efficientnetb0", "efficientnet"),
}


def _normalise_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def find_checkpoints(directory=None) -> dict[str, Path]:
    """Map every known architecture to a ``.pth/.pt/.ckpt`` file if present."""
    folder = Path(directory) if directory else DEFAULT_CHECKPOINTS_DIR
    found: dict[str, Path] = {}

    if not folder.exists():
        return found

    files = sorted(
        [*folder.glob("*.pth"), *folder.glob("*.pt"), *folder.glob("*.ckpt")],
        key=lambda p: p.name.lower(),
    )

    for model_name, patterns in _FILE_PATTERNS.items():
        for path in files:
            stem = _normalise_key(path.stem)
            if any(pattern in stem for pattern in patterns):
                found[model_name] = path
                break

    return found


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older PyTorch without the ``weights_only`` argument.
        return torch.load(path, map_location=map_location)
    except Exception:
        # Checkpoint contains non-tensor objects -> allow full unpickling.
        return torch.load(path, map_location=map_location, weights_only=False)


def _unwrap_state_dict(raw):
    if not isinstance(raw, dict):
        return raw
    for key in ("state_dict", "model_state_dict", "model", "weights"):
        value = raw.get(key)
        if isinstance(value, dict):
            return value
    return raw


def load_checkpoint(
    model_name: str,
    checkpoint_path,
    device: str = "cpu",
    dropout: float = 0.3,
    num_classes: int = 3,
):
    """Build an architecture and load a checkpoint as robustly as possible.

    Handles raw state dicts, ``{"state_dict": ...}`` wrappers, DataParallel
    ``module.`` prefixes and torchvision wrappers saved with or without the
    outer ``model.`` prefix.  Keys whose name/shape do not match are skipped
    and reported instead of crashing the app.

    Returns ``(model, report)`` with the model already on ``device`` in
    eval mode.
    """
    model = build_model(model_name, num_classes=num_classes, dropout=dropout,
                        pretrained=False)

    raw = _torch_load(str(checkpoint_path), map_location="cpu")
    state = _unwrap_state_dict(raw)

    cleaned = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    if not cleaned:
        raise ValueError(f"No tensor entries found in {checkpoint_path}")

    model_sd = model.state_dict()
    target_keys = set(model_sd)

    candidates = {
        "as_is": cleaned,
        "add_model_prefix": {
            key if key.startswith("model.") else f"model.{key}": value
            for key, value in cleaned.items()
        },
        "strip_model_prefix": {
            key[len("model."):] if key.startswith("model.") else key: value
            for key, value in cleaned.items()
        },
    }

    strategy, best = max(
        candidates.items(),
        key=lambda item: len(target_keys & set(item[1])),
    )

    adapted, skipped = {}, []
    for key, value in best.items():
        if key in model_sd and tuple(model_sd[key].shape) == tuple(value.shape):
            adapted[key] = value
        else:
            skipped.append(key)

    missing = [key for key in model_sd if key not in adapted]

    model.load_state_dict(adapted, strict=False)
    model.to(device)
    model.eval()

    report = {
        "checkpoint": str(checkpoint_path),
        "strategy": strategy,
        "matched": len(adapted),
        "total": len(model_sd),
        "missing": missing,
        "skipped": skipped,
        "complete": not missing and not skipped,
    }
    return model, report


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def _sync(device) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def predict(model, image_bgr, model_name: str, device="cpu", normalization=None):
    """Run the full inference pipeline on a BGR frame.

    Returns a dict with the predicted label, per-class probabilities, pure
    network latency (ms), the face bounding box and the cropped face.
    """
    norm = normalization or NORMALIZATION.get(model_name, "minus1_1")

    rgb, box, face_detected = detect_and_crop(image_bgr)
    tensor = to_tensor(rgb, norm).unsqueeze(0).to(device)

    _sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        _sync(device)
    infer_ms = (time.perf_counter() - start) * 1000.0

    probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    index = int(np.argmax(probs))

    return {
        "label": CLASS_NAMES[index],
        "index": index,
        "confidence": float(probs[index]),
        "probs": [float(p) for p in probs],
        "infer_ms": infer_ms,
        "box": box,
        "face_detected": face_detected,
        "face_rgb": rgb,
    }


def decision_for(label: str, confidence: float, threshold: float) -> str:
    """Map a prediction to the access decision shown in the demo."""
    if confidence < threshold:
        return "UNCERTAIN - low confidence"
    if label == "Real":
        return "GRANTED - genuine face"
    if label == "Print":
        return "DENIED - print attack"
    return "DENIED - replay attack"


# --------------------------------------------------------------------------- #
# Efficiency benchmark
# --------------------------------------------------------------------------- #
def benchmark(model, device="cpu", runs: int = 20, warmup: int = 5) -> dict:
    """Measure pure forward-pass latency/FPS of an architecture.

    Weights are irrelevant to the cost of a forward pass, so the benchmark can
    run on randomly initialised models (no checkpoint required).
    """
    model.eval()
    x = torch.randn(1, 3, 224, 224, device=device)

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            model(x)
        _sync(device)

        times = []
        for _ in range(max(1, runs)):
            start = time.perf_counter()
            model(x)
            _sync(device)
            times.append((time.perf_counter() - start) * 1000.0)

    latency = float(np.mean(times))
    deviation = float(np.std(times))

    return {
        "params": count_parameters(model),
        "size_mb": round(model_size_mb(model), 2),
        "latency_ms": round(latency, 2),
        "latency_std_ms": round(deviation, 2),
        "fps": round(1000.0 / latency, 1) if latency > 0 else 0.0,
        "device": str(device),
    }
