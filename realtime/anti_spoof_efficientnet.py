# ==========================================
# anti_spoof_efficientnet.py
# ==========================================
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "checkpoints", "EfficientNetB0_S1_v2_best.pth")

CLASS_NAMES = ["REAL", "PRINT SPOOF", "DIGITAL SCREEN SPOOF"]   # urutan = label training (0, 1, 2)

IMAGE_SIZE = 224
DROPOUT = 0.3
MARGIN = 0.30

# Normalisasi ImageNet (sama dengan resnet_eval_transform di training)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# LOAD EFFICIENTNETB0
# ==========================================

def load_efficientnet():
    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, 3)
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # Hapus prefiks "model." dari kelas pembungkus EfficientNetB0Model
    state_dict = {
        (k.replace("model.", "", 1) if k.startswith("model.") else k): v
        for k, v in checkpoint.items()
    }

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print("EfficientNetB0 loaded successfully!")
    print("Device:", device)
    return model


model = load_efficientnet()


# ==========================================
# FACE PREPROCESSING (sama dengan training)
# ==========================================

def crop_square(image, box, margin=MARGIN):
    """Crop persegi di sekitar wajah, selalu di dalam gambar (tanpa padding)."""
    x, y, w, h = box
    H, W = image.shape[:2]

    side = min(int(max(w, h) * (1 + 2 * margin)), H, W)

    x1 = int(round(x + w / 2 - side / 2))
    y1 = int(round(y + h / 2 - side / 2))
    x1 = min(max(0, x1), W - side)
    y1 = min(max(0, y1), H - side)

    return image[y1:y1 + side, x1:x1 + side]


def preprocess_face(frame, face_box, margin=MARGIN):
    x, y, w, h = map(int, face_box)

    crop = crop_square(frame, (x, y, w, h), margin)
    if crop.size == 0:
        return None, None

    interp = cv2.INTER_AREA if crop.shape[0] > IMAGE_SIZE else cv2.INTER_LINEAR
    crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    arr = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    return tensor, rgb


# ==========================================
# PREDICT FACE
# ==========================================

def predict_face(frame, face_box):
    input_tensor, rgb = preprocess_face(frame, face_box)

    if input_tensor is None:
        return None

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = (time.perf_counter() - start_time) * 1000

    predicted_class = int(probabilities.argmax())

    return {
        "class": CLASS_NAMES[predicted_class],
        "class_id": predicted_class,
        "confidence": float(probabilities[predicted_class]),   # 0-1, dikali 100 di realtime
        "probabilities": probabilities.tolist(),
        "inference_time": inference_time,
        "crop_rgb": rgb,                                        # untuk debug (tekan S)
    }

# import os
# import time

# import cv2
# import torch
# import torch.nn as nn

# from PIL import Image
# from torchvision import models, transforms


# # ==========================================
# # CONFIGURATION
# # ==========================================

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MODEL_PATH = os.path.join(
#     BASE_DIR,
#     "models",
#     "EfficientNetB0_S1_v2_best.pth"
# )

# CLASS_NAMES = [
#     "REAL",
#     "PRINT SPOOF",
#     "DIGITAL SCREEN SPOOF"
# ]

# IMAGE_SIZE = 224
# DROPOUT = 0.3

# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )


# # ==========================================
# # PREPROCESSING
# # ==========================================

# transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])



# # ==========================================
# # LOAD EFFICIENTNETB0
# # ==========================================

# def load_efficientnet():

#     model = models.efficientnet_b0(weights=None)

#     in_features = model.classifier[1].in_features

#     model.classifier = nn.Sequential(
#         nn.Dropout(p=DROPOUT),
#         nn.Linear(in_features, 3)
#     )

#     checkpoint = torch.load(
#         MODEL_PATH,
#         map_location=device,
#         weights_only=True
#     )

#     if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#         checkpoint = checkpoint["state_dict"]

#     state_dict = {}

#     for key, value in checkpoint.items():

#         new_key = (
#             key.replace("model.", "", 1)
#             if key.startswith("model.")
#             else key
#         )

#         state_dict[new_key] = value

#     model.load_state_dict(state_dict)

#     model = model.to(device)
#     model.eval()

#     print("EfficientNetB0 loaded successfully!")
#     print("Device:", device)

#     return model


# model = load_efficientnet()


# # ==========================================
# # FACE PREPROCESSING
# # ==========================================

# def preprocess_face(frame, face_box, margin=0.30):

#     x, y, w, h = map(int, face_box)

#     height, width = frame.shape[:2]

#     x1 = max(0, int(x - w * margin))
#     y1 = max(0, int(y - h * margin))

#     x2 = min(width, int(x + w * (1 + margin)))
#     y2 = min(height, int(y + h * (1 + margin)))

#     face = frame[y1:y2, x1:x2]

#     if face.size == 0:
#         return None

#     face_rgb = cv2.cvtColor(
#         face,
#         cv2.COLOR_BGR2RGB
#     )

#     face_pil = Image.fromarray(face_rgb)

#     input_tensor = transform(face_pil)

#     input_tensor = input_tensor.unsqueeze(0).to(device)

#     return input_tensor


# # ==========================================
# # PREDICT FACE
# # ==========================================

# def predict_face(frame, face_box):

#     input_tensor = preprocess_face(
#         frame,
#         face_box
#     )

#     if input_tensor is None:
#         return None

#     if device.type == "cuda":
#         torch.cuda.synchronize()

#     start_time = time.perf_counter()

#     with torch.no_grad():

#         outputs = model(input_tensor)

#         probabilities = torch.softmax(
#             outputs,
#             dim=1
#         )

#     if device.type == "cuda":
#         torch.cuda.synchronize()

#     inference_time = (
#         time.perf_counter() - start_time
#     ) * 1000

#     probabilities = probabilities[0].cpu().numpy()

#     predicted_class = int(probabilities.argmax())

#     return {
#         "class": CLASS_NAMES[predicted_class],
#         "class_id": predicted_class,
#         "confidence": float(probabilities[predicted_class]),
#         "probabilities": probabilities.tolist(),
#         "inference_time": inference_time
#     }
