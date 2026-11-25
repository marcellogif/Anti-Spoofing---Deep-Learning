# ==========================================
# realtime_efficientnet.py
# ==========================================
import time
from collections import deque

import cv2
import numpy as np

from anti_spoof_efficientnet import predict_face, CLASS_NAMES


# ==================================
# CONFIG
# ==================================

MIN_FACE = 110        # wajah lebih kecil dari ini diabaikan (di-upscale jadi buram)
MAX_FACES = 1         # proses wajah terbesar saja (ubah ke 2-3 jika perlu)
SMOOTH_FRAMES = 8     # rata-rata probabilitas dari N frame terakhir
CONF_THRESHOLD = 70   # di bawah ini ditampilkan "UNCERTAIN"

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))   # sama dengan preprocessing training

history = deque(maxlen=SMOOTH_FRAMES)
save_id = 0


# ==================================
# WEBCAM
# ==================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot access webcam!")

print("EfficientNetB0 Real-Time Detection Started")
print("Q = keluar | S = simpan frame & crop untuk debug")

prev_time = time.perf_counter()


# ==================================
# LOOP
# ==================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    gray = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE, MIN_FACE)
    )

    # Urutkan dari terbesar, ambil MAX_FACES
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:MAX_FACES]

    if len(faces) == 0:
        history.clear()

    for face_box in faces:
        x, y, w, h = map(int, face_box)

        result = predict_face(frame, face_box)
        if result is None:
            continue

        # Smoothing probabilitas
        history.append(result["probabilities"])
        probs = np.mean(history, axis=0) * 100
        class_id = int(probs.argmax())
        confidence = float(probs[class_id])

        if confidence < CONF_THRESHOLD:
            label, color = "UNCERTAIN", (0, 165, 255)
        elif class_id == 0:
            label, color = CLASS_NAMES[0], (0, 200, 0)
        else:
            label, color = CLASS_NAMES[class_id], (0, 0, 255)

        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

        cv2.putText(display_frame, f"{label} ({confidence:.1f}%)",
                    (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.putText(display_frame,
                    f"R {probs[0]:.0f} | P {probs[1]:.0f} | D {probs[2]:.0f}",
                    (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(display_frame, f"Inference: {result['inference_time']:.1f} ms",
                    (x, y + h + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # FPS
    now = time.perf_counter()
    fps = 1 / max(now - prev_time, 1e-9)
    prev_time = now
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("EfficientNetB0 Real-Time Anti-Spoofing", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite(f"debug_frame_{save_id}.png", frame)          # frame mentah
        for k, fb in enumerate(faces):
            r = predict_face(frame, fb)
            if r:
                cv2.imwrite(f"debug_crop_{save_id}_{k}.png",
                            cv2.cvtColor(r["crop_rgb"], cv2.COLOR_RGB2BGR))
        print(f"Disimpan: debug_frame_{save_id}.png")
        save_id += 1


# ==================================
# CLEANUP
# ==================================

cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")