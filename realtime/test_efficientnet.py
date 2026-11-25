
import cv2
import tkinter as tk

from tkinter import filedialog
from anti_spoof_efficientnet import predict_face


# ==================================
# SELECT IMAGE
# ==================================

root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select Face Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp")
    ]
)

if not image_path:
    print("No image selected!")
    exit()


# ==================================
# LOAD IMAGE
# ==================================

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Cannot read selected image!")

print("Image:", image_path)


# ==================================
# FACE DETECTION
# ==================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(60, 60)
)

if len(faces) == 0:
    print("No face detected!")
    exit()


# ==================================
# PREDICTION
# ==================================

for index, face_box in enumerate(faces):

    x, y, w, h = face_box

    result = predict_face(
        image,
        face_box
    )

    if result is None:
        continue

    print("\n==============================")
    print(f"FACE {index + 1}")
    print("==============================")

    print("Prediction:", result["class"])

    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        "Inference Time:",
        f"{result['inference_time']:.2f} ms"
    )

    for name, probability in zip(
        [
            "REAL",
            "PRINT SPOOF",
            "DIGITAL SCREEN SPOOF"
        ],
        result["probabilities"]
    ):
        print(f"{name}: {probability * 100:.2f}%")

    # Draw bounding box
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        result["class"],
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


# ==================================
# DISPLAY RESULT
# ==================================

cv2.imshow(
    "EfficientNetB0 Anti-Spoofing",
    image
)

cv2.waitKey(0)
cv2.destroyAllWindows()
