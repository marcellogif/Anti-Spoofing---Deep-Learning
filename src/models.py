"""
Model zoo for the Face Anti-Spoofing comparison (COMP6826001).

Four architectures are compared on the same 3-class problem:

    Index 0 : Real          (genuine face)
    Index 1 : Print attack  (photo printed on paper)
    Index 2 : Replay attack (photo/video shown on a screen)

- Custom CNN      : baseline, trained from scratch.
- MobileNetV2     : transfer learning, tuned for edge/real-time inference.
- ResNet18        : transfer learning with residual connections.
- EfficientNetB0  : transfer learning with compound scaling.

Every transfer-learning backbone is initialised with ImageNet weights and
frozen during training; only the final classification head is updated
(exactly as in the training notebook).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CLASS_NAMES = ["Real", "Print", "Replay"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

# Order used throughout the UI / report tables.
MODEL_ORDER = ["Custom CNN", "MobileNetV2", "ResNet18", "EfficientNetB0"]

# Normalisation MUST match the one used during training.
#   minus1_1  -> pixels scaled to [-1, 1]  (Custom CNN, cf. proposal section 2)
#   imagenet   -> ImageNet mean/std        (backbones pretrained on ImageNet)
NORMALIZATION = {
    "Custom CNN": "minus1_1",
    "MobileNetV2": "imagenet",
    "ResNet18": "imagenet",
    "EfficientNetB0": "imagenet",
}

MODEL_DESCRIPTIONS = {
    "Custom CNN": {
        "role": "Baseline - trained from scratch",
        "summary": (
            "Four convolutional blocks (32 > 64 > 128 > 256 channels), each with "
            "Conv2d, BatchNorm, ReLU and MaxPool, followed by global average "
            "pooling, dropout (0.3) and a linear classifier."
        ),
        "strength": "Light-weight reference point; no pretrained weights required.",
        "origin": "Custom architecture (this project)",
    },
    "MobileNetV2": {
        "role": "Transfer learning - ImageNet",
        "summary": (
            "Depthwise-separable convolutions with inverted residuals and linear "
            "bottlenecks. The backbone is frozen and the classifier head is "
            "replaced for 3 classes."
        ),
        "strength": "Smallest and fastest - designed for real-time / on-device use.",
        "origin": "Sandler et al., 2018 (torchvision)",
    },
    "ResNet18": {
        "role": "Transfer learning - ImageNet",
        "summary": (
            "18-layer residual network with skip connections that ease gradient "
            "flow. Backbone frozen, final fully-connected layer replaced."
        ),
        "strength": "Proven deep feature extractor; strong accuracy baseline.",
        "origin": "He et al., 2016 (torchvision)",
    },
    "EfficientNetB0": {
        "role": "Transfer learning - ImageNet",
        "summary": (
            "Compound-scaled network (depth, width and resolution scaled "
            "uniformly) with squeeze-and-excitation blocks. Head replaced for "
            "3 classes."
        ),
        "strength": "Usually the best accuracy/compute trade-off of the four.",
        "origin": "Tan & Le, 2019 (torchvision)",
    },
}


# --------------------------------------------------------------------------- #
# Architectures
# --------------------------------------------------------------------------- #
class CustomCNN(nn.Module):
    """Baseline CNN used as the performance reference (Skenario 1 notebook)."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class ResNet18Model(nn.Module):
    """ResNet18 with a frozen backbone and a new 3-class head."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3,
                 pretrained: bool = False):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.model = models.resnet18(weights=weights)

        # Freeze backbone (mirrors the training notebook).
        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MobileNetV2Model(nn.Module):
    """MobileNetV2 with a frozen backbone and a new 3-class head."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3,
                 pretrained: bool = False):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.model = models.mobilenet_v2(weights=weights)

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EfficientNetB0Model(nn.Module):
    """EfficientNet-B0 with a frozen backbone and a new 3-class head."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3,
                 pretrained: bool = False):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_model(name: str, num_classes: int = NUM_CLASSES, dropout: float = 0.3,
                pretrained: bool = False) -> nn.Module:
    """Factory for the four compared architectures.

    ``pretrained=False`` by default so that the app never downloads ImageNet
    weights: a trained checkpoint overwrites every weight anyway, and the
    benchmark only measures architecture speed.
    """
    name = (name or "").strip()

    if name in ("Custom CNN", "CustomCNN", "CNN"):
        return CustomCNN(num_classes=num_classes, dropout=dropout)
    if name in ("ResNet18", "ResNet-18", "resnet18"):
        return ResNet18Model(num_classes=num_classes, dropout=dropout,
                             pretrained=pretrained)
    if name in ("MobileNetV2", "MobileNet-V2", "mobilenet_v2"):
        return MobileNetV2Model(num_classes=num_classes, dropout=dropout,
                                pretrained=pretrained)
    if name in ("EfficientNetB0", "EfficientNet-B0", "efficientnet_b0"):
        return EfficientNetB0Model(num_classes=num_classes, dropout=dropout,
                                   pretrained=pretrained)

    raise ValueError(
        f"Unknown model '{name}'. Choose one of: {', '.join(MODEL_ORDER)}"
    )


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Total (or trainable) parameter count."""
    return sum(
        p.numel()
        for p in model.parameters()
        if (p.requires_grad or not trainable_only)
    )


def model_size_mb(model: nn.Module) -> float:
    """Approximate float32 checkpoint size in megabytes."""
    return count_parameters(model) * 4.0 / (1024 ** 2)
