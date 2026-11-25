import os
import cv2
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


# ==========================================
# DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "checkpoints", "ResNet18_best.pth")

CLASS_NAMES = [
    "REAL",
    "PRINT SPOOF",
    "DIGITAL SCREEN SPOOF"
]


# ==========================================
# RESNET18 ARCHITECTURE
# ==========================================

model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(512, 3)
)


# ==========================================
# LOAD MODEL
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

# Remove "model." prefix if it exists
new_state_dict = {}

for key, value in checkpoint.items():
    if key.startswith("model."):
        new_key = key.replace("model.", "", 1)
    else:
        new_key = key

    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)

model = model.to(device)
model.eval()

print("ResNet18 loaded successfully!")
print("Device:", device)


# ==========================================
# IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# FACE PREPROCESSING
# ==========================================

def preprocess_face(frame, face_box):

    x, y, w, h = face_box

    margin = 0.30

    x1 = max(0, int(x - w * margin))
    y1 = max(0, int(y - h * margin))
    x2 = min(frame.shape[1], int(x + w * (1 + margin)))
    y2 = min(frame.shape[0], int(y + h * (1 + margin)))

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return None

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = Image.fromarray(face)

    face = transform(face)
    face = face.unsqueeze(0).to(device)

    return face


# ==========================================
# PREDICTION
# ==========================================

def predict_face(frame, face_box):

    face_tensor = preprocess_face(frame, face_box)

    if face_tensor is None:
        raise ValueError("Invalid face crop!")

    with torch.no_grad():
        outputs = model(face_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    class_id = predicted_class.item()

    return {
        "class_id": class_id,
        "label": CLASS_NAMES[class_id],
        "confidence": float(confidence.item() * 100),
        "probabilities": (
            probabilities[0].cpu().numpy() * 100
        ).tolist()
    }
