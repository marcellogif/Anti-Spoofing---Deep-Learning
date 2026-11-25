"""
Face Anti-Spoofing — Comparative Analysis of Deep Learning Models
Streamlit demo application for COMP6826001 (Deep Learning).

Run with:

    streamlit run app.py

Pages
-----
1. Overview          — problem, objectives, scope and headline results
2. Dataset           — NUAA + Face Anti-Spoofing Large Dataset, split, pipeline
3. Models & Scenarios— the four architectures and the hyper-parameter grid
4. Results           — metrics, per-class breakdown and live efficiency benchmark
5. Live Demo         — real-time webcam / image-upload anti-spoofing inference
"""

from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.inference import (
    benchmark,
    decision_for,
    find_checkpoints,
    load_checkpoint,
    predict,
)
from src.models import (
    CLASS_NAMES,
    MODEL_ORDER,
    MODEL_DESCRIPTIONS,
    NORMALIZATION,
    build_model,
    count_parameters,
    model_size_mb,
)
from src.preprocessing import annotate_frame

try:  # Plotly gives nicer charts, but the app must not break without it.
    import plotly.express as px

    HAS_PLOTLY = True
except Exception:  # pragma: no cover
    HAS_PLOTLY = False

# --------------------------------------------------------------------------- #
# Page config / constants
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Face Anti-Spoofing | COMP6826001",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "results" / "results.json"
CHECKPOINTS_DIR = ROOT / "checkpoints"

PAGE_OVERVIEW = "🏠 Overview"
PAGE_DATASET = "📚 Dataset"
PAGE_MODELS = "🧠 Models & Scenarios"
PAGE_RESULTS = "📈 Results"
PAGE_DEMO = "🎥 Live Demo"
PAGES = [PAGE_OVERVIEW, PAGE_DATASET, PAGE_MODELS, PAGE_RESULTS, PAGE_DEMO]

AUTHORS = [
    ("Marcello Y. H. Manurung", "2802557333"),
    ("Marcell Ilinatino", "2802482111"),
    ("Jason Brandon Loi", "2802484584"),
]

st.markdown(
    """
    <style>
    .fas-card {
        border: 1px solid #e8e8e8; border-radius: 14px; padding: 18px 20px;
        background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
        height: 100%;
    }
    .fas-card h4 { margin: 0 0 4px 0; }
    .fas-badge {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 700; background: #eef3ff; color: #2c3e66;
        white-space: nowrap;
    }
    .fas-src {
        border: 1px solid #e8e8e8; border-radius: 14px; padding: 16px 18px;
        background: #ffffff; height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    defaults = {
        "live": False,
        "pred_history": collections.deque(maxlen=10),
        "active_model": None,
        "e2e_ema": None,
        "infer_ema": None,
        "benchmark": None,
        "use_gpu": torch.cuda.is_available(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# --------------------------------------------------------------------------- #
# Cached resources / data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_results() -> dict:
    with open(RESULTS_FILE, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner=False)
def arch_stats(name: str) -> tuple[int, float]:
    """Parameter count and float32 size of an architecture (no weights)."""
    model = build_model(name, pretrained=False)
    return count_parameters(model), round(model_size_mb(model), 2)


@st.cache_resource(show_spinner="⏳ Loading model weights…")
def load_weights(model_name: str, checkpoint: str, device: str, dropout: float = 0.3):
    return load_checkpoint(model_name, checkpoint, device=device, dropout=dropout)


@st.cache_resource(show_spinner="⏳ Building architecture (random weights)…")
def random_model(model_name: str, device: str, dropout: float = 0.3):
    model = build_model(model_name, dropout=dropout, pretrained=False)
    model.to(device)
    model.eval()
    return model


def resolve_device() -> str:
    if torch.cuda.is_available() and st.session_state.get("use_gpu", True):
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------- #
# Webcam capture (module-level state survives Streamlit re-runs)
# --------------------------------------------------------------------------- #
_CAPTURES: dict[int, "cv2.VideoCapture"] = {}


def get_capture(index: int) -> "cv2.VideoCapture":
    capture = _CAPTURES.get(index)
    if capture is None or not capture.isOpened():
        capture = cv2.VideoCapture(index)
        _CAPTURES[index] = capture
    return capture


def release_captures() -> None:
    for capture in _CAPTURES.values():
        try:
            capture.release()
        except Exception:
            pass
    _CAPTURES.clear()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🛡️ Face Anti-Spoofing")
        st.caption("**COMP6826001** · Deep Learning · Final Project")

        page = st.radio("Navigation", PAGES, label_visibility="collapsed")

        st.divider()
        st.markdown("**Group members**")
        for name, student_id in AUTHORS:
            st.markdown(f"- **{name}**  \n  `{student_id}`")

        st.divider()
        gpu_available = torch.cuda.is_available()
        st.checkbox(
            "Use GPU (CUDA)",
            disabled=not gpu_available,
            key="use_gpu",
            help="Uncheck to force CPU inference.",
        )
        if gpu_available:
            st.caption(f"🖥️ `{torch.cuda.get_device_name(0)}`")
        else:
            st.caption("🖥️ CUDA not available — running on CPU.")

        st.divider()
        st.caption(
            "Checkpoints are read from the `checkpoints/` folder "
            "(`CustomCNN_best.pth`, `ResNet18_best.pth`, …)."
        )

    return page


# --------------------------------------------------------------------------- #
# Page 1 — Overview
# --------------------------------------------------------------------------- #
def page_overview(results: dict) -> None:
    completed = [r for r in results["results"] if r.get("status") == "completed"]
    best = max(completed, key=lambda r: r["accuracy"]) if completed else None
    total_images = results["dataset"]["merged"]["total"]

    st.title("🛡️ Face Anti-Spoofing Detection")
    st.subheader(
        "Comparative Analysis of Deep Learning Models for Presentation Attack "
        "Detection in Facial Recognition Systems"
    )

    st.markdown(
        " | ".join(f"**{name}** (`{sid}`)" for name, sid in AUTHORS)
    )
    st.caption(
        f"Course **{results['meta']['course']} — {results['meta']['course_name']}**"
        " · Face anti-spoofing final project"
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Architectures compared", len(MODEL_ORDER))
    c2.metric("Attack classes", "3")
    c3.metric("Images (merged dataset)", f"{total_images:,}")
    c4.metric(
        "Best test accuracy",
        f"{best['accuracy']:.2%}" if best else "—",
    )

    st.divider()

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("### 1 · Problem")
        st.markdown(
            """
            Facial-recognition systems are widely used for attendance, access
            control and financial authentication, yet they remain vulnerable to
            **presentation attacks**: showing a printed photo (*print attack*) or
            a photo/video displayed on a screen (*replay attack*) to the camera.

            In December 2025 several public servants in Wenzhou, China were
            caught using printed photos of colleagues to clock in for them
            (South China Morning Post, 2025), and similar weaknesses in
            face-based attendance systems have been identified in Indonesia
            (Wisesa dkk., 2025). Failed presentation-attack detection directly
            harms data integrity, institutional accountability and trust in
            biometric systems (Xing dkk., 2025).

            Print and replay attacks leave only **visual traces** — paper
            texture, moiré patterns and screen reflections — which CNNs can
            learn automatically, but each architecture offers a different
            accuracy / speed trade-off (Najeebullah dkk., 2025). A systematic
            comparison is therefore required before real-time deployment.
            """
        )

    with right:
        st.markdown("### 2 · Objectives")
        st.markdown(
            """
            1. **Analyse and compare** the performance of several Deep Learning
               architectures on 3-class face anti-spoofing classification
               (*Real*, *Print*, *Replay*).
            2. **Identify the architecture** with the best balance between
               detection accuracy and inference speed for real-time deployment.
            3. **Implement the best model** in a webcam-based attendance
               simulation and evaluate it live (latency, FPS, accuracy).

            **Scope** — print and replay presentation attacks only.
            Deepfake and injection (digital) attacks are out of scope.
            """
        )

        st.markdown("### 3 · Method at a glance")
        st.markdown(
            """
            - **Data**: NUAA (print) + Face Anti-Spoofing Large Dataset (replay)
            - **Pipeline**: Haar Cascade face detection > 30 % margin crop >
              RGB > 224×224 > normalisation
            - **Models**: Custom CNN (baseline), MobileNetV2, ResNet18,
              EfficientNetB0 (ImageNet transfer learning)
            - **Metrics**: accuracy, precision, recall, F1, confusion matrix,
              inference latency, FPS and model size
            """
        )

    st.divider()
    st.markdown("### Headline results (Scenario 1)")
    st.caption(
        "Values read directly from the executed training notebook "
        "`notebooks/Anti_Spoofing_Skenario_1.ipynb`."
    )
    if best:
        st.success(
            f"**{best['model']}** currently leads with "
            f"**{best['accuracy']:.2%} test accuracy** "
            f"(F1 {best['f1']:.4f}) on {best['test_support']:,} test images."
        )


# --------------------------------------------------------------------------- #
# Page 2 — Dataset
# --------------------------------------------------------------------------- #
def page_dataset(results: dict) -> None:
    st.header("📚 Dataset, Split & Preprocessing")
    ds = results["dataset"]

    # --- sources ------------------------------------------------------------ #
    cols = st.columns(2, gap="large")
    for col, src in zip(cols, ds["sources"]):
        with col:
            st.markdown(
                f"""
                <div class="fas-src">
                    <h4>{src['name']}</h4>
                    <p style="margin:4px 0 10px 0; color:#666; font-size:0.85rem;">
                        Kaggle: <code>{src['kaggle']}</code>
                    </p>
                    <p style="margin:0;">
                        <b>Total:</b> {src['total']:,} images<br>
                        <b>Real:</b> {src['real']:,} &nbsp;·&nbsp;
                        <b>Spoof:</b> {src['spoof']:,}<br>
                        <b>Attack type:</b> {src['attack_type']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "Both datasets are merged so that the two physical presentation "
        "attacks we detect are covered: **print (paper)** from NUAA and "
        "**replay (screen)** from the Face Anti-Spoofing Large Dataset."
    )

    # --- merged distribution ------------------------------------------------ #
    st.markdown("#### Merged class distribution")
    merged = ds["merged"]
    dist = pd.DataFrame(
        [
            {"Class": "Real (genuine)", "Images": merged["classes"]["Real"], "Label": 0},
            {"Class": "Print attack", "Images": merged["classes"]["Print"], "Label": 1},
            {"Class": "Replay attack", "Images": merged["classes"]["Replay"], "Label": 2},
        ]
    )

    col_chart, col_table = st.columns([3, 2], gap="large")
    with col_chart:
        if HAS_PLOTLY:
            fig = px.bar(
                dist, x="Class", y="Images", color="Class", text="Images",
                color_discrete_sequence=["#2ecc71", "#e74c3c", "#f39c12"],
            )
            fig.update_layout(showlegend=False, yaxis_title="Images")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(dist.set_index("Class")["Images"])
    with col_table:
        st.dataframe(
            dist.style.format({"Images": "{:,}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"**Total: {merged['total']:,} images** (label mapping "
                   "Real = 0, Print = 1, Replay = 2)")

    # --- splits ------------------------------------------------------------- #
    st.markdown("#### Train / validation / test split")
    splits = ds["splits"]
    after = ds["after_preprocessing"]

    split_rows = []
    for split, ratio, key in (
        ("Training", "70 %", "train"),
        ("Validation", "15 %", "validation"),
        ("Testing", "15 %", "test"),
    ):
        before = splits[key]
        classes = after["by_split"][key]
        split_rows.append(
            {
                "Split": split,
                "Ratio": ratio,
                "Before preprocessing": before,
                "After preprocessing": sum(classes.values()),
                "Real": classes["Real"],
                "Print": classes["Print"],
                "Replay": classes["Replay"],
            }
        )

    split_df = pd.DataFrame(split_rows)
    st.dataframe(
        split_df.style.format(
            {
                "Before preprocessing": "{:,}",
                "After preprocessing": "{:,}",
                "Real": "{:,}",
                "Print": "{:,}",
                "Replay": "{:,}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Stratified 70/15/15 split (seed 42). After Haar-Cascade preprocessing "
        f"**{after['total_processed']:,}** images were kept and "
        f"**{after['failed']}** were discarded because no face was detected."
    )

    # --- pipeline ----------------------------------------------------------- #
    st.markdown("#### Preprocessing & augmentation pipeline")
    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.markdown(
            """
            **Preprocessing (train + inference, identical):**

            1. Face detection with **OpenCV Haar Cascade**
               (`frontalface_default`, fallback `frontalface_alt2`) on the
               histogram-equalised grayscale image
            2. Largest face cropped with a **30 % margin** to keep context
            3. **BGR → RGB** conversion
            4. Resize to **224 × 224** pixels
            5. Convert to float32 and normalise:
               - Custom CNN → **[-1, 1]**
               - Transfer-learning models → **ImageNet** mean/std
            6. Label encoding: **Real = 0, Print = 1, Replay = 2**
            """
        )
    with p2:
        st.markdown(
            """
            **Augmentation (training only):**

            - Random horizontal flip (p = 0.5)
            - Random rotation ±15°
            - Color jitter (brightness ±0.2)

            **Anti-leakage note:** where subject/session information is
            available the split is performed per identity or per recording
            session, so frames of the same subject never appear in both the
            training and the testing set.
            """
        )


# --------------------------------------------------------------------------- #
# Page 3 — Models & scenarios
# --------------------------------------------------------------------------- #
def render_model_card(name: str, checkpoints: dict[str, Path]) -> None:
    desc = MODEL_DESCRIPTIONS[name]
    params, size_mb = arch_stats(name)
    checkpoint = checkpoints.get(name)
    status = f"✅ `{checkpoint.name}`" if checkpoint else "⏳ not found yet"

    st.markdown(
        f"""
        <div class="fas-card">
            <div style="display:flex; justify-content:space-between;
                        align-items:center; gap:8px;">
                <h4 style="margin:0;">{name}</h4>
                <span class="fas-badge">{desc['role']}</span>
            </div>
            <p style="margin:10px 0 6px 0;">{desc['summary']}</p>
            <p style="margin:0 0 8px 0;"><b>Why this model:</b> {desc['strength']}</p>
            <p style="margin:0; font-size:0.84rem; color:#666; line-height:1.5;">
                Origin: {desc['origin']}<br>
                Parameters: <b>{params:,}</b> ({size_mb:.2f} MB float32)<br>
                Normalisation: <code>{NORMALIZATION[name]}</code><br>
                Checkpoint: {status}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_models(results: dict, checkpoints: dict[str, Path]) -> None:
    st.header("🧠 Selected Models & Experiment Scenarios")
    st.markdown(
        """
        Four architectures are compared on the **same** 3-class problem with the
        **same** data split and evaluation metrics, so the comparison is fair.
        MobileNetV2, ResNet18 and EfficientNetB0 start from **ImageNet
        pretrained weights** with a **frozen backbone**; only the final
        classification head is trained (transfer learning + fine-tuning of the
        head). Custom CNN is the **baseline trained from scratch**.
        """
    )

    for start in (0, 2):
        cols = st.columns(2, gap="large")
        for col, name in zip(cols, MODEL_ORDER[start:start + 2]):
            with col:
                render_model_card(name, checkpoints)

    st.divider()
    st.markdown("#### Hyper-parameter scenarios")
    scenario_rows = []
    for s in results["scenarios"]:
        scenario_rows.append(
            {
                "Scenario": f"Scenario {s['id']}",
                "Learning rate": f"{s['learning_rate']:g}",
                "Batch size": s["batch_size"],
                "Optimizer": s["optimizer"],
                "Epochs": s["epochs"],
                "Dropout": s["dropout"],
                "Momentum": (
                    f"{s['momentum']}" if s["momentum"] is not None else "—"
                ),
            }
        )
    st.dataframe(pd.DataFrame(scenario_rows), use_container_width=True,
                 hide_index=True)

    st.info(
        "**Scenario 1** (Custom CNN + ResNet18) has been executed — see the "
        "Results page. Scenarios 2–4 cover MobileNetV2 / EfficientNetB0 runs "
        "and will be appended to `results/results.json` once training "
        "notebooks finish. Every scenario is evaluated on the identical split "
        "and metrics for an objective comparison."
    )


# --------------------------------------------------------------------------- #
# Page 4 — Results
# --------------------------------------------------------------------------- #
def page_results(results: dict, checkpoints: dict[str, Path]) -> None:
    st.header("📈 Evaluation Results")
    all_results = results["results"]
    completed = [r for r in all_results if r.get("status") == "completed"]
    pending = [r["model"] for r in all_results if r.get("status") != "completed"]

    if pending:
        st.info(
            "⏳ **Awaiting training runs:** " + ", ".join(pending)
            + " — update `results/results.json` when those notebooks finish."
        )

    if not completed:
        st.warning("No completed runs found in `results/results.json` yet.")
        return

    # --- headline ----------------------------------------------------------- #
    best = max(completed, key=lambda r: r["accuracy"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best accuracy", f"{best['accuracy']:.2%}")
    m2.metric("Best F1 (macro)", f"{max(r['f1'] for r in completed):.4f}")
    m3.metric(
        "Best validation accuracy",
        f"{max(r['val_accuracy'] for r in completed):.2%}",
    )
    m4.metric(
        "Fastest training epoch",
        f"{min(r['avg_epoch_time_s'] for r in completed):.1f} s",
    )

    # --- metric table + chart ---------------------------------------------- #
    st.markdown("#### Test-set metrics (15 % hold-out)")
    metric_df = pd.DataFrame(
        [
            {
                "Model": r["model"],
                "Scenario": r["scenario"],
                "Accuracy": round(r["accuracy"], 4),
                "Precision (macro)": round(r["precision"], 4),
                "Recall (macro)": round(r["recall"], 4),
                "F1-Score (macro)": round(r["f1"], 4),
                "Test images": r["test_support"],
            }
            for r in completed
        ]
    )
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    if HAS_PLOTLY:
        long_df = metric_df.melt(
            id_vars=["Model"],
            value_vars=["Accuracy", "Precision (macro)", "Recall (macro)",
                        "F1-Score (macro)"],
            var_name="Metric",
            value_name="Score",
        )
        long_df["label"] = long_df["Score"].map(lambda v: f"{v:.3f}")
        fig = px.bar(
            long_df, x="Model", y="Score", color="Metric", barmode="group",
            text="label", range_y=[0, 1.08],
            color_discrete_sequence=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="Score", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    # --- per-class ---------------------------------------------------------- #
    st.markdown("#### Per-class report (test set)")
    for r in completed:
        if not r.get("per_class"):
            continue
        with st.expander(f"📋 {r['model']} — per-class precision / recall / F1"):
            rows = [
                {
                    "Class": cls,
                    "Precision": vals["precision"],
                    "Recall": vals["recall"],
                    "F1-Score": vals["f1"],
                    "Support": vals["support"],
                }
                for cls, vals in r["per_class"].items()
            ]
            st.dataframe(
                pd.DataFrame(rows).style.format(
                    {"Precision": "{:.2f}", "Recall": "{:.2f}",
                     "F1-Score": "{:.2f}", "Support": "{:,}"}
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Per-class values are the rounded figures printed by "
                "`classification_report` in the notebook; full-precision "
                "aggregates are in the table above."
            )

    # --- training summary --------------------------------------------------- #
    st.markdown("#### Training summary (best epoch by validation loss)")
    summary_df = pd.DataFrame(
        [
            {
                "Model": r["model"],
                "Scenario": r["scenario"],
                "Best epoch": r["best_epoch"],
                "Train loss": round(r["train_loss"], 4),
                "Val loss": round(r["val_loss"], 4),
                "Train accuracy": round(r["train_accuracy"], 4),
                "Val accuracy": round(r["val_accuracy"], 4),
                "Avg epoch time (s)": round(r["avg_epoch_time_s"], 2),
            }
            for r in completed
        ]
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.caption(
        "Confusion matrices for both models are available in the executed "
        "cells of `notebooks/Anti_Spoofing_Skenario_1.ipynb`."
    )

    st.divider()

    # --- efficiency benchmark ---------------------------------------------- #
    st.markdown("#### ⚙️ Efficiency benchmark (measured on this machine)")
    st.caption(
        results["efficiency"]["note"]
        + "  \nLatency below is a **pure forward pass** (batch = 1, 224×224); "
          "weights do not affect inference cost, so architectures without a "
          "trained checkpoint can be measured too."
    )

    if st.button("▶ Run benchmark (4 architectures)", type="primary"):
        device = resolve_device()
        progress = st.progress(0.0, text="Benchmarking…")
        rows = []
        for i, name in enumerate(MODEL_ORDER):
            model = build_model(name, pretrained=False).to(device).eval()
            stats = benchmark(model, device, runs=20, warmup=5)

            checkpoint = checkpoints.get(name)
            ckpt_mb = (
                round(checkpoint.stat().st_size / (1024 ** 2), 2)
                if checkpoint else None
            )
            rows.append(
                {
                    "Model": name,
                    "Parameters": f"{stats['params']:,}",
                    "Size (MB)": stats["size_mb"],
                    "Checkpoint (MB)": f"{ckpt_mb:.2f}" if ckpt_mb is not None else "—",
                    "Latency (ms)": stats["latency_ms"],
                    "± std (ms)": stats["latency_std_ms"],
                    "FPS": stats["fps"],
                }
            )
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            progress.progress(
                (i + 1) / len(MODEL_ORDER), text=f"Benchmarked {name}"
            )
        progress.empty()
        st.session_state.benchmark = {"device": device, "rows": rows}

    if st.session_state.benchmark:
        bench = st.session_state.benchmark
        st.caption(
            f"Device: `{bench['device']}` · "
            f"{torch.cuda.get_device_name(0) if bench['device'] == 'cuda' else 'CPU'}"
            f" · measured {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        st.dataframe(pd.DataFrame(bench["rows"]), use_container_width=True,
                     hide_index=True)

        st.markdown("**Latency types reported in this project**")
        st.markdown(
            """
            - **Network inference latency** — forward pass only (table above).
            - **End-to-end pipeline latency** — webcam capture + Haar Cascade
              + preprocessing + inference + rendering; shown live on the
              *Live Demo* page together with the effective FPS.
            """
        )

    st.caption(results["efficiency"]["iso_standard"])


# --------------------------------------------------------------------------- #
# Page 5 — Live demo
# --------------------------------------------------------------------------- #
def _update_ema(key: str, value: float, alpha: float = 0.25) -> float:
    previous = st.session_state.get(key)
    if previous is None:
        st.session_state[key] = value
    else:
        st.session_state[key] = alpha * value + (1 - alpha) * previous
    return st.session_state[key]


def _live_frame(model, model_name: str, device: str, threshold: float,
                cam_index: int, mirror: bool, smooth: bool) -> None:
    """Grab a single frame, run the pipeline, render it, then re-run."""
    capture = get_capture(cam_index)

    if not capture.isOpened():
        st.error("❌ Could not open the webcam — check the camera index / permissions.")
        st.session_state.live = False
        release_captures()
        return

    ok, frame = capture.read()
    if not ok or frame is None:
        st.error("❌ Failed to read a frame from the webcam.")
        st.session_state.live = False
        release_captures()
        return

    if mirror:
        frame = cv2.flip(frame, 1)

    start = time.perf_counter()
    result = predict(model, frame, model_name=model_name, device=device)

    if smooth:
        history = st.session_state.pred_history
        history.append(result["index"])
        display_index = collections.Counter(history).most_common(1)[0][0]
    else:
        display_index = result["index"]

    display_label = CLASS_NAMES[display_index]
    display_conf = result["probs"][display_index]
    status = decision_for(display_label, display_conf, threshold)

    vis = annotate_frame(
        frame,
        result["box"],
        display_label,
        display_conf,
        status=status,
        face_detected=result["face_detected"],
    )
    e2e_ms = (time.perf_counter() - start) * 1000.0

    e2e_ema = _update_ema("e2e_ema", e2e_ms)
    _update_ema("infer_ema", result["infer_ms"])
    fps = 1000.0 / e2e_ema if e2e_ema > 0 else 0.0

    video_col, info_col = st.columns([2.4, 1], gap="large")

    with video_col:
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_column_width=True)

    with info_col:
        if status.startswith("GRANTED"):
            st.success(f"✅ ACCESS **GRANTED** — genuine face")
        elif status.startswith("DENIED"):
            st.error(f"🚫 ACCESS **DENIED** — {display_label} attack")
        else:
            st.warning("⚠️ **UNCERTAIN** — confidence below threshold")

        st.metric("Predicted class", display_label)
        st.metric("Confidence", f"{display_conf:.1%}")
        st.metric("Network latency", f"{result['infer_ms']:.1f} ms")
        st.metric("End-to-end frame", f"{e2e_ema:.1f} ms (EMA)")
        st.metric("Effective FPS", f"{fps:.1f}")
        st.caption(
            "Face detected: "
            + ("✅ yes" if result["face_detected"]
               else "❌ no — full frame used as fallback")
        )
        st.caption(
            "Per-class probability — "
            + " · ".join(
                f"{name} {p:.0%}"
                for name, p in zip(CLASS_NAMES, result["probs"])
            )
        )

    # Cap the loop around ~30 FPS so the browser/CPU is not saturated.
    remaining = (1.0 / 30.0) - (time.perf_counter() - start)
    if remaining > 0:
        time.sleep(remaining)

    st.rerun()


def page_live_demo(results: dict, checkpoints: dict[str, Path]) -> None:
    st.header("🎥 Live Demo — Real-time Face Anti-Spoofing")

    # checkpoint status cards
    status_cols = st.columns(len(MODEL_ORDER))
    for col, name in zip(status_cols, MODEL_ORDER):
        checkpoint = checkpoints.get(name)
        with col:
            if checkpoint:
                st.success(f"✅ **{name}**\n\n`{checkpoint.name}`")
            else:
                st.info(f"⏳ **{name}**\n\nno checkpoint")

    if not checkpoints:
        st.warning(
            """
            **No trained checkpoints found.** Train the models in
            `notebooks/Anti_Spoofing_Skenario_1.ipynb` (Colab), download the
            `*_best.pth` files and drop them into the `checkpoints/`
            folder of this repository:

            ```
            checkpoints/
            ├── CustomCNN_best.pth
            ├── ResNet18_best.pth
            ├── MobileNetV2_best.pth
            └── EfficientNetB0_best.pth
            ```

            Only tick *"Use random weights"* below if you want to test the
            interface — random weights give meaningless predictions.
            """
        )

    allow_random = st.checkbox(
        "Use random weights instead of a trained checkpoint (interface test only)",
        value=False,
    )

    if allow_random:
        options = list(MODEL_ORDER)
    else:
        options = [name for name in MODEL_ORDER if name in checkpoints]

    if not options:
        st.info("Upload at least one checkpoint to enable the demo.")
        return

    control_left, control_right = st.columns([2, 1], gap="large")
    with control_left:
        model_name = st.selectbox("Model", options)
    with control_right:
        threshold = st.slider(
            "Confidence threshold", 0.50, 0.99, 0.70, 0.01,
            help="Predictions below this confidence are shown as UNCERTAIN.",
        )

    if st.session_state.get("active_model") != model_name:
        st.session_state.active_model = model_name
        st.session_state.pred_history.clear()
        st.session_state.e2e_ema = None
        st.session_state.infer_ema = None

    device = resolve_device()

    try:
        if allow_random:
            model = random_model(model_name, device)
        else:
            model, report = load_weights(
                model_name, str(checkpoints[model_name]), device
            )
            if not report["complete"] and (
                report["missing"] or report["skipped"]
            ):
                st.warning(
                    f"Checkpoint `{Path(report['checkpoint']).name}` loaded "
                    f"partially ({report['matched']}/{report['total']} tensors). "
                    f"Missing: {len(report['missing'])}, skipped: "
                    f"{len(report['skipped'])} — make sure the checkpoint was "
                    f"trained for 3 classes with this exact architecture."
                )
    except Exception as exc:  # pragma: no cover
        st.error(f"Could not load model **{model_name}**: `{exc}`")
        return

    st.caption(
        "Pipeline: **webcam → Haar Cascade → 30 % margin crop → RGB → "
        "224×224 → normalise → inference → overlay** "
        f"(normalisation: `{NORMALIZATION[model_name]}`, device: `{device}`)"
    )

    mode = st.radio("Input mode", ["📷 Live webcam", "🖼️ Upload image"],
                    horizontal=True)

    if mode.startswith("📷"):
        cam_left, cam_right = st.columns([3, 1])
        with cam_left:
            cam_index = int(
                st.number_input("Camera index", 0, 5, 0, step=1)
            )
        with cam_right:
            st.write("")
            mirror = st.checkbox("Mirror", value=True)

        smooth = st.checkbox(
            "Smooth predictions (majority of last 10 frames)", value=True
        )

        btn_start, btn_stop, _ = st.columns([1, 1, 2])
        with btn_start:
            start_clicked = st.button(
                "▶ Start", type="primary", disabled=st.session_state.live
            )
        with btn_stop:
            stop_clicked = st.button(
                "⏹ Stop", disabled=not st.session_state.live
            )

        if start_clicked:
            st.session_state.live = True
            st.session_state.pred_history.clear()
            st.session_state.e2e_ema = None
            st.session_state.infer_ema = None

        if stop_clicked:
            st.session_state.live = False
            release_captures()

        if st.session_state.live:
            _live_frame(
                model=model,
                model_name=model_name,
                device=device,
                threshold=threshold,
                cam_index=cam_index,
                mirror=mirror,
                smooth=smooth,
            )
        else:
            st.info(
                "Press **▶ Start** to open the webcam. Hold a real face, a "
                "printed photo or a phone/tablet screen 40–60 cm from the "
                "camera to test the three classes."
            )

        with st.expander("🧪 User-testing protocol (5 participants)"):
            st.caption(
                "Three lighting conditions (bright, dim, backlight), "
                "40–60 cm from the camera, 3 attempts per cell."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Scenario": "Genuine face", "Attack media": "—",
                         "Trials / condition": 3},
                        {"Scenario": "Print attack", "Attack media":
                         "Printed photo, eye area cut out, held in front of the face",
                         "Trials / condition": 3},
                        {"Scenario": "Print attack (mask)", "Attack media":
                         "Printed photo worn as a mask", "Trials / condition": 3},
                        {"Scenario": "Replay attack (photo)", "Attack media":
                         "Photo displayed on a phone / tablet screen",
                         "Trials / condition": 3},
                        {"Scenario": "Replay attack (video)", "Attack media":
                         "Video played on a phone / tablet screen",
                         "Trials / condition": 3},
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    else:
        uploaded = st.file_uploader(
            "Upload a face image (real / print / screen photo)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        if uploaded is None:
            st.info("Upload an image to run a single prediction.")
            return

        buffer = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Could not decode that image.")
            return

        result = predict(model, image, model_name=model_name, device=device)
        status = decision_for(result["label"], result["confidence"], threshold)
        vis = annotate_frame(
            image, result["box"], result["label"], result["confidence"],
            status=status, face_detected=result["face_detected"],
        )

        img_col, info_col = st.columns([2.4, 1], gap="large")
        with img_col:
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                     use_column_width=True, caption=uploaded.name)
        with info_col:
            if status.startswith("GRANTED"):
                st.success("✅ ACCESS **GRANTED** — genuine face")
            elif status.startswith("DENIED"):
                st.error(f"🚫 ACCESS **DENIED** — {result['label']} attack")
            else:
                st.warning("⚠️ **UNCERTAIN** — confidence below threshold")

            st.metric("Predicted class", result["label"])
            st.metric("Confidence", f"{result['confidence']:.1%}")
            st.metric("Network latency", f"{result['infer_ms']:.1f} ms")
            st.metric(
                "Implied FPS",
                f"{1000.0 / result['infer_ms']:.1f}"
                if result["infer_ms"] > 0 else "—",
            )
            st.caption(
                "Face detected: "
                + ("✅ yes" if result["face_detected"]
                   else "❌ no — full image used as fallback")
            )

        st.markdown("#### Class probabilities")
        for name, prob in zip(CLASS_NAMES, result["probs"]):
            st.progress(float(prob), text=f"**{name}** — {prob:.1%}")


# --------------------------------------------------------------------------- #
# Footer / entry point
# --------------------------------------------------------------------------- #
def render_footer() -> None:
    st.divider()
    st.caption(
        "COMP6826001 · Deep Learning · "
        + " · ".join(name for name, _ in AUTHORS)
        + " · Face Anti-Spoofing — print & replay presentation attacks"
    )


def main() -> None:
    results = load_results()
    checkpoints = find_checkpoints(CHECKPOINTS_DIR)

    page = render_sidebar()

    # Leaving the demo page must not leave the camera running.
    if page != PAGE_DEMO and st.session_state.live:
        st.session_state.live = False
        release_captures()

    if page == PAGE_OVERVIEW:
        page_overview(results)
    elif page == PAGE_DATASET:
        page_dataset(results)
    elif page == PAGE_MODELS:
        page_models(results, checkpoints)
    elif page == PAGE_RESULTS:
        page_results(results, checkpoints)
    else:
        page_live_demo(results, checkpoints)

    render_footer()


# Streamlit executes the whole file on every interaction, so ``main()`` runs
# unconditionally (no ``if __name__ == "__main__"`` guard needed).
main()
