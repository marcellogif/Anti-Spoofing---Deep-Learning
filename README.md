# Comparative Analysis of Deep Learning Models for Face Anti-Spoofing Detection in Facial Recognition Systems

**Course:** `COMP6826001` — Deep Learning
**Project:** Final Project — Computer Vision / Face Anti-Spoofing

| Group member | Student ID |
| --- | --- |
| Marcello Y. H. Manurung | 2802557333 |
| Marcell Ilinatino | 2802482111 |
| Jason Brandon Loi | 2802484584 |

This repository contains the training notebook and the real-time Streamlit
demo used to train, compare and deploy four Deep Learning architectures for
**print** and **replay** presentation-attack detection.

---

## Table of Contents

1. [Problem](#1-problem)
2. [Objectives](#2-objectives)
3. [Dataset Source and Description](#3-dataset-source-and-description)
4. [Selected Deep Learning Methods and Baseline](#4-selected-deep-learning-methods-and-baseline)
5. [Planned Evaluation Metrics](#5-planned-evaluation-metrics)
6. [Results So Far](#6-results-so-far)
7. [Repository Structure](#7-repository-structure)
8. [Getting Started](#8-getting-started)
9. [Deployment & User Testing Plan](#9-deployment--user-testing-plan)
10. [References](#10-references)

---

## 1. Problem

Facial-recognition systems are widely deployed for identity verification —
attendance (presensi), access control and financial-service authentication.
However, they remain vulnerable to **presentation attacks**: fooling the camera
with fake face artefacts such as a printed photograph (*print attack*) or a
photo/video shown on a phone or tablet screen (*replay attack*).

Physical presentation attacks are the most realistic to carry out in the real
world because they are performed directly in front of the camera. A face
recognition system without an anti-spoofing mechanism therefore cannot
guarantee its users' security (Xing dkk., 2025).

Real-world incidents show the impact:

- In **December 2025**, several public servants in Wenzhou, China were found
  using printed photos of their colleagues' faces to trick face-based
  attendance machines, allowing one person to clock in for multiple absent
  employees ("China public servants," 2025).
- In **Indonesia**, similar vulnerabilities were identified in face-based
  attendance systems that can be manipulated with photos or videos
  (Wisesa dkk., 2025).

Failed presentation-attack detection directly harms **data integrity,
institutional accountability and trust in biometric systems**.

This research focuses on **print** and **replay** attacks, since both are the
most frequently encountered physical presentation attacks and remain the focus
of anti-spoofing research (Xing dkk., 2025). Both attacks leave only *visual
traces* — paper texture, moiré patterns and screen reflections — which are hard
to distinguish from a genuine face. CNN architectures are chosen because they
learn such visual traces automatically, but each architecture differs
significantly in performance and inference speed (Najeebullah dkk., 2025), so a
systematic comparison is needed to select the model best suited for real-time
deployment. **Deepfake and injection (digital) attacks are out of scope.**

### Research questions

1. How do various Deep Learning architectures perform at detecting face
   spoofing (**print** and **replay** attacks) in a facial recognition system?
2. Which architecture offers the best balance between anti-spoofing detection
   accuracy and inference speed for real-time deployment?

### Objectives

1. Analyse and compare the performance of several Deep Learning architectures
   for 3-class face anti-spoofing classification (**Real**, **Print**,
   **Replay**).
2. Implement and test the selected model in a webcam-based real-time testing
   system.

---

## 2. Dataset Source and Description

Two datasets are downloaded from **Kaggle** and merged, because the spoof types
we detect come from different sources: **print (paper)** from NUAA and
**replay (screen)** from the Face Anti-Spoofing Large Dataset.

| Dataset | Kaggle source | Total | Real | Spoof | Attack type |
| --- | --- | ---: | ---: | ---: | --- |
| **NUAA** | `aleksandrpikul222/nuaaaa` | 12,614 | 5,105 | 7,509 | Print attack (photo on paper) |
| **Face Anti-Spoofing Large Dataset** | `mdhasibulhossain231/face-anti-spoofing-large-dataset-image-video` | 8,112 | 3,832 | 4,280 | Replay attack (screen / photo / video) |
| **Merged** | — | **20,726** | **8,937** | **11,789** | print + replay |

### NUAA

The NUAA dataset contains face images labelled as *real* (client) and *spoof*
(imposter). The portion used in this project consists of **12,614 images**:
**5,105 genuine** and **7,509 spoof** (print) images.

### Face Anti-Spoofing Large Dataset

This dataset contains **8,112 images** — **3,832 real** and **4,280 spoof** —
mostly PNG and JPEG/JPG with varying resolutions. It provides image *and*
video folders; only the **image** folder is used, and inside it only the
`Real` and `Spoof` subfolders.

### Splitting, preprocessing and augmentation

The merged data is split **70 % training / 15 % validation / 15 % testing**
(stratified, `random_state=42`):

| Split | Ratio | Images (before preprocessing) |
| --- | ---: | ---: |
| Training | 70 % | 14,508 |
| Validation | 15 % | 3,109 |
| Testing | 15 % | 3,109 |

To prevent **data leakage**, the split is performed per identity subject or
per recording session whenever that information is available, so frames from
the same subject/session never appear in both the training and the test set.

**Preprocessing pipeline** (identical at training and inference time):

1. **Face detection** with OpenCV **Haar Cascade**
   (`haarcascade_frontalface_default`, fallback `haarcascade_frontalface_alt2`)
   on a histogram-equalised grayscale image.
2. **Face cropping** of the largest detected face with a **30 % margin** to
   keep surrounding visual context.
3. **BGR → RGB** conversion.
4. **Resizing** to **224 × 224** pixels.
5. Conversion to **float32** and **normalisation to [-1, 1]** (Custom CNN);
   transfer-learning models use ImageNet mean/std so it matches their
   pretrained weights.
6. Numeric labelling: **Real = 0, Print = 1, Replay = 2**
   (`Digital_Screen_Spoof` folder ≡ Replay class).

**Augmentation (training only):** random horizontal flip (p = 0.5), random
rotation ±15°, color jitter (brightness ±0.2).

After preprocessing, **20,627 images** were retained and **99 images** were
discarded because no face could be detected:

| Split | Real | Print | Replay | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 6,244 | 5,249 | 2,939 | 14,432 |
| validation | 1,339 | 1,124 | 638 | 3,101 |
| test | 1,339 | 1,123 | 632 | 3,094 |

---

## 3. Selected Deep Learning Methods and Baseline

### 3.1 Selected models

The project compares **four** architectures for the 3-class anti-spoofing
task:

| Model | Role | Idea |
| --- | --- | --- |
| **Custom CNN** | Baseline | 4 conv blocks (32 → 64 → 128 → 256) with BatchNorm/ReLU/MaxPool, global average pooling and a dropout classifier; trained **from scratch** to establish a reference performance. |
| **MobileNetV2** | Efficiency candidate | Depthwise-separable convolutions with inverted residuals; evaluated for its computational efficiency and suitability for real-time webcam deployment. |
| **ResNet18** | Deep feature candidate | Residual connections; used to investigate how effective residual learning is for extracting discriminative face features. |
| **EfficientNetB0** | Trade-off candidate | Compound-scaled (depth/width/resolution) architecture; evaluated for its balance between classification performance and computational cost. |

All four models are trained to classify face images into **Real**, **Print**
and **Replay**, and are compared using the *same* data split and evaluation
metrics to guarantee a fair comparison.

### 3.2 Transfer learning and fine-tuning

MobileNetV2, ResNet18 and EfficientNetB0 are initialised with **ImageNet
pretrained weights** to help extract generic image features. The final
classification layer of each model is replaced so it outputs 3 classes, and
each model is then trained on the research dataset to adapt its features to
distinguishing the three classes. The backbones are **frozen** during
training — only the new classification head is updated (the training notebook
follows exactly this scheme).

### 3.3 Experiment scenarios and hyper-parameter configuration

| Hyper-parameter | Scenario 1 | Scenario 2 | Scenario 3 | Scenario 4 |
| --- | --- | --- | --- | --- |
| Learning rate | 0.001 | 0.0001 | 0.00001 | 0.0001 |
| Batch size | 32 | 16 | 16 | 32 |
| Optimizer | Adam | Adam | Adam | SGD |
| Epochs | 10 | 20 | 30 | 40 |
| Dropout | 0.3 | 0.2 | 0.5 | 0.3 |
| Momentum | – | – | – | 0.9 |

Every scenario is evaluated with the same dataset split and metrics so that
the effect of the hyper-parameters on the 3-class classification performance
can be compared objectively. **Scenario 1 has been executed** (Custom CNN and
ResNet18 — see [Results](#6-results-so-far)); the remaining runs are pending.

---

## 4. Planned Evaluation Metrics

| Group | Metrics | Purpose |
| --- | --- | --- |
| General classification | **Accuracy, Precision, Recall, F1-score** (macro + per class) | Overall performance and balance between classes. |
| Error analysis | **Confusion matrix** | Detailed prediction distribution per category (Real / Print / Replay). |
| Biometric standard | **ISO/IEC 30107-3:2023** (APCER / BPCER) | Standard presentation-attack detection reporting, planned for the user-testing phase. |
| Computational efficiency | **Inference time (ms), FPS, Model size (MB), Latency** | Pure network forward-pass latency vs. **end-to-end webcam pipeline latency** (capture + face detection + preprocessing + display). |

---

## 5. Results So Far

Scenario 1 (Custom CNN vs. ResNet18, LR 0.001, batch 32, Adam, 10 epochs,
dropout 0.3) — values taken directly from the executed cells of
[`Anti_Spoofing_Skenario_1.ipynb`](Anti_Spoofing_Skenario_1.ipynb) on the
3,094-image test set:

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) | Best epoch | Val accuracy | Avg epoch time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Custom CNN** | **0.9664** | **0.9554** | **0.9674** | **0.9607** | 9 | 0.9700 | 61.61 s |
| ResNet18 | 0.9470 | 0.9439 | 0.9293 | 0.9358 | 10 | 0.9487 | 56.22 s |

Per-class report (test set):

| Model | Class | Precision | Recall | F1 | Support |
| --- | --- | ---: | ---: | ---: | ---: |
| Custom CNN | Real | 0.98 | 0.95 | 0.96 | 1,339 |
| Custom CNN | Print | 1.00 | 0.99 | 0.99 | 1,123 |
| Custom CNN | Replay (Digital Screen Spoof) | 0.89 | 0.97 | 0.92 | 632 |
| ResNet18 | Real | 0.92 | 0.96 | 0.94 | 1,339 |
| ResNet18 | Print | 0.99 | 0.99 | 0.99 | 1,123 |
| ResNet18 | Replay (Digital Screen Spoof) | 0.92 | 0.83 | 0.87 | 632 |

**Observation:** the from-scratch Custom CNN currently outperforms the
ImageNet-frozen ResNet18 on this merged print/replay dataset, while ResNet18
trains slightly faster per epoch. MobileNetV2 and EfficientNetB0 runs are
pending; all results live in [`results/results.json`](results/results.json),
which the Streamlit app renders.

---

## 6. Repository Structure

```
DeepLearning/
├── Anti_Spoofing_Skenario_1.ipynb   # Training notebook (Colab) — Skenario 1
├── app.py                           # Streamlit application (entry point)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── src/
│   ├── models.py                    # CustomCNN, ResNet18, MobileNetV2, EfficientNetB0
│   ├── preprocessing.py             # Haar Cascade, crop, resize, normalisation, overlay
│   └── inference.py                 # Checkpoint loading, prediction, benchmark
│
├── results/
│   └── results.json                 # Dataset stats, scenarios and evaluation metrics
│
└── checkpoints/                     # Trained *.pth weights (not committed)
    └── README.md                    # Expected file names & how to obtain them
```

---

## 7. Getting Started

### 7.1 Requirements

- Python **3.9+**
- Optional: an NVIDIA GPU (CUDA) for faster training/inference — the app falls
  back to CPU automatically.

### 7.2 Training (Google Colab)

1. Open [`Anti_Spoofing_Skenario_1.ipynb`](Anti_Spoofing_Skenario_1.ipynb) in
   Google Colab.
2. Run the **Download File** cells — they pull both Kaggle datasets:

   ```bash
   kaggle datasets download aleksandrpikul222/nuaaaa
   kaggle datasets download mdhasibulhossain231/face-anti-spoofing-large-dataset-image-video
   ```

3. Run the remaining cells in order: dataset merging → configuration →
   splitting → preprocessing (Haar Cascade) → training → evaluation.
4. Download the best checkpoints at the end of the notebook:

   ```python
   from google.colab import files
   files.download("/content/model_checkpoints/CustomCNN_best.pth")
   files.download("/content/model_checkpoints/ResNet18_best.pth")
   ```

### 7.3 Running the Streamlit demo (local machine)

#### Quick start (dependencies already installed)

Open a terminal in the project folder and run:

```powershell
cd D:\Documents\Code\Binus\DeepLearning
streamlit run app.py
```

The app opens automatically in the browser at `http://localhost:8501`.
Stop the server with **Ctrl + C** in the terminal.

#### Full setup (fresh machine / virtual environment)

```bash
git clone <this-repository>
cd DeepLearning

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

Then open the URL shown in the terminal (default
`http://localhost:8501`).

> **Port already in use?** Run `streamlit run app.py --server.port 8502`
> (or any other free port).

#### Webcam checklist (Live Demo page)

1. **Must be served from `localhost`** (or HTTPS) — browsers only grant
   camera access to `localhost`/HTTPS, so opening the app via a network IP
   will not expose the webcam.
2. Go to the **🎥 Live Demo** page, pick the model, then press **▶ Start**
   (**⏹ Stop** to release the camera).
3. If the camera does not open, change the **Camera index** and try `0`, `1`
   or `2`.
4. Test the three classes at 40–60 cm from the camera:
   - **Real** — your genuine face → `ACCESS GRANTED`
   - **Print** — a printed photo (eye area cut out) held in front of the face
   - **Replay** — a photo or video of the face shown on a phone/tablet screen
5. The panel shows the predicted class, confidence score, network latency,
   end-to-end frame time and effective FPS.

#### Before the demo — checklist

1. Put the trained checkpoints in [`checkpoints/`](checkpoints/README.md)
   (see [§7.4](#74-adding-trained-weights)) — otherwise the Live Demo page
   will tell you exactly what to do.
2. Optionally tick **"Use random weights"** to exercise the interface only —
   predictions with random weights are meaningless and are clearly marked.
3. Prepare the attack media (printed photo, phone/tablet with photo/video)
   and test under bright, dim and backlight conditions.

### 7.4 Adding trained weights

Copy the `*_best.pth` files produced by the notebook into
[`checkpoints/`](checkpoints/README.md):

```
checkpoints/
├── CustomCNN_best.pth
├── ResNet18_best.pth
├── MobileNetV2_best.pth
└── EfficientNetB0_best.pth
```

The app detects them automatically (any `*.pth`/`*.pt`/`*.ckpt` whose name
contains `customcnn`, `mobilenet`, `resnet18` or `efficientnet`). Without a
checkpoint the *Live Demo* page explains what to do and can still be exercised
in a clearly-marked *random weights* mode.

### 7.5 App pages

| Page | Contents |
| --- | --- |
| 🏠 **Overview** | Problem, objectives, scope and headline results |
| 📚 **Dataset** | Sources, class distribution, split, preprocessing & augmentation |
| 🧠 **Models & Scenarios** | The four architectures + hyper-parameter scenario table |
| 📈 **Results** | Metrics tables/charts, per-class reports, live efficiency benchmark (parameters, MB, ms, FPS) |
| 🎥 **Live Demo** | Real-time webcam or uploaded-image inference with bounding box, label, confidence and access decision |

---

## 8. Deployment & User Testing Plan

### 8.1 System implementation

The best-performing model is deployed into a face-based attendance simulation
written in **Python** with **OpenCV**. The pipeline has five stages:

1. Frame acquisition from the webcam;
2. Face detection with **Haar Cascade**;
3. Preprocessing identical to training — 30 % margin face crop, BGR → RGB,
   resize to 224 × 224, normalisation to [-1, 1];
4. Model inference;
5. Visualisation — bounding box, predicted label (**Real**, **Print**,
   **Replay**) and confidence score.

Hardware specifications of the test machine are recorded so that speed
measurements are reproducible.

### 8.2 User testing plan

Five consenting participants are tested under **three lighting conditions**
(bright, dim, backlight) at a distance of **40–60 cm** from the camera,
3 attempts per cell:

| Scenario | Attack media | Trials per condition |
| --- | --- | ---: |
| Genuine face | – | 3 |
| Print attack | Printed photo with the eye area cut out, held in front of the face | 3 |
| Print attack (mask) | Printed photo worn as a mask | 3 |
| Replay attack (photo) | Participant's photo displayed on a phone/tablet screen | 3 |
| Replay attack (video) | Participant's video played on a phone/tablet screen | 3 |

Real-time evaluation covers **latency/FPS** and detection accuracy in the
actual environment, separating *network inference latency* from *end-to-end
pipeline latency* (capture, face detection, preprocessing, rendering).

---

## 9. References

- China public servants use face masks to bypass facial recognition machines,
  skip work. 2025. *South China Morning Post.*
  <https://www.scmp.com/news/people-culture/trending-china/article/3335029/china-public-servants-use-face-masks-bypass-facial-recognition-help-each-other-skip-work>

- International Organization for Standardization. (2023). *Information
  technology — Biometric presentation attack detection — Part 3: Testing and
  reporting (ISO/IEC 30107-3:2023).* <https://www.iso.org/standard/79520.html>

- Najeebullah, Salman, M., & Swati, Z. N. K. 2025. *Face spoofing detection
  using deep learning.* <https://doi.org/10.48550/arXiv.2503.19223>

- Xing, H., Tan, S. Y., Qamar, F., & Jiao, Y. 2025. *Face anti-spoofing based
  on deep learning: A comprehensive survey.* Applied Sciences.
  <https://doi.org/10.3390/app15126891>

- Wisesa, B. A., Putri, V. M., Faristasari, E., & Duli, S. A. J. 2025.
  *Preventive attendance record using photo from mobile phone and printed
  paper using CNN.* J-INTECH.
  <https://jurnal.stiki.ac.id/J-INTECH/article/view/1927>

- Salehi, A. W., Khan, S., Gupta, G., Alabduallah, B. I., Almjally, A.,
  Alsolai, H., ... & Mellit, A. (2023). *A study of CNN and transfer learning
  in medical imaging: Advantages, challenges, future scope.* Sustainability,
  15(7), 5930. <https://www.mdpi.com/2071-1050/15/7/5930>

- Zhu, Q., Zhuang, H., Zhao, M., Xu, S., & Meng, R. (2024). *A study on
  expression recognition based on improved MobileNetV2 network.* Scientific
  Reports, 14(1), 8121.
  <https://www.nature.com/articles/s41598-024-58736-x.pdf>

- Liu, H., Brailsford, T., & Bull, L. (2024, October). *ResNet18 performance:
  Impact of network depth and image resolution on image classification.* In
  Proceedings of the 2024 8th International Conference on Advances in
  Artificial Intelligence (pp. 351–356).
  <https://dl.acm.org/doi/pdf/10.1145/3704137.3704173>

- Alshammari, H. H., Mahmood, M. A., & Elbashir, M. K. (2025). *Explainable
  fusion of EfficientNetB0 and ResNet50 for liver fibrosis staging in
  ultrasound imaging.* Scientific Reports, 16(1), 3536.
  <https://www.nature.com/articles/s41598-025-33544-z.pdf>

---

*COMP6826001 — Deep Learning · Marcello Y. H. Manurung (2802557333) ·
Marcell Ilinatino (2802482111) · Jason Brandon Loi (2802484584)*
