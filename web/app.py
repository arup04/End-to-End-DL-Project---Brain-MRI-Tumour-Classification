import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image

# Ensure the package is importable by adding src directory to system path
web_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(web_dir, "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from brain_mri.config.configuration import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    IMAGE_SIZE,
    DEVICE,
    MODEL_PATH,
    MODEL_NAME,
)
from brain_mri.inference.predictor import (
    load_model_weights,
    preprocess_image,
    predict,
)
from brain_mri.inference.gradcam import (
    GradCAM,
    get_target_layer,
    generate_heatmap,
    create_overlay,
)

# ── Badge colors for each class ──────────────────────────────────────────────
CLASS_BADGE_COLORS = {
    "glioma":     ("#3B82F6", "🟦"),   # blue
    "meningioma": ("#F59E0B", "🟨"),   # amber
    "notumor":    ("#10B981", "🟩"),   # green
    "pituitary":  ("#EF4444", "🟥"),   # red
}


# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_resource
def load_cached_model():
    """Load model and create a cached GradCAM instance."""
    try:
        model = load_model_weights()
        target_layer = get_target_layer(model)
        gradcam = GradCAM(model, target_layer)
        return model, gradcam
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


@st.cache_data
def get_project_summary():
    """Load training summary from artifacts/logs/project_summary.json."""
    summary_path = os.path.abspath(
        os.path.join(web_dir, "..", "artifacts", "logs", "project_summary.json")
    )
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ── CSS ──────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    /* ── Header ── */
    .hero-title {
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #1E88E5 0%, #7C4DFF 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center !important;
        margin-bottom: 0.1rem !important;
        letter-spacing: -0.5px;
        line-height: 1.1 !important;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #9e9e9e;
        text-align: center;
        margin-bottom: 2rem;
        line-height: 1.6;
    }

    /* ── Navigation buttons ── */
    div[data-testid="stHorizontalBlock"] > div > div > button[kind="secondary"] {
        border-radius: 12px;
    }



    /* ── Prediction card ── */
    .pred-card {
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(30,136,229,0.08), rgba(124,77,255,0.08));
        border: 1px solid rgba(124,77,255,0.15);
    }
    .pred-label {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .pred-conf {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #10B981, #1E88E5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ── Probability bar ── */
    .prob-bar-outer {
        background: #2a2a2a;
        border-radius: 8px;
        height: 24px;
        margin: 4px 0 8px 0;
        overflow: hidden;
    }
    .prob-bar-inner {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease;
    }
    .prob-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.9rem;
    }

    /* ── Class badges (sidebar) ── */
    .class-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 3px 2px;
    }

    /* ── Footer ── */
    .footer-wrapper {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #777;
    }
    .footer-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #1E88E5, #7C4DFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .footer-tech {
        font-size: 0.85rem;
        color: #999;
        margin-bottom: 0.2rem;
    }
    .footer-author {
        font-size: 0.8rem;
        color: #666;
    }
</style>
"""


# ── Page view: Scan Analysis ─────────────────────────────────────────────────
def render_scan_analysis(model, gradcam, heatmap_sensitivity, heatmap_sharpness):
    """Render the MRI scan upload, prediction, and Grad-CAM explanation UI."""

    # Upload section
    st.markdown("### 📤 Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Upload an MRI scan",
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI image (JPG, JPEG, or PNG)",
        label_visibility="collapsed",
    )
    st.caption("Supported · **PNG** · **JPG** · **JPEG**")

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")

    # Auto-analyze on upload
    with st.spinner("🤖 AI is analyzing the scan…"):
        try:
            tensor = preprocess_image(image)
            res = predict(model, tensor)
            pred_idx = res["class_index"]
            pred_label = res["class_name"]
            confidence = res["confidence"]
            probabilities = res["probabilities"]
            display_name = DISPLAY_NAMES[pred_label]

            heatmap = generate_heatmap(
                gradcam, tensor, pred_idx, image, gamma=heatmap_sharpness
            )
            overlay = create_overlay(image, heatmap, threshold=heatmap_sensitivity)
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            st.exception(e)
            return

    # ── Layout: Image | Prediction ────────────────────────────────────────
    col_img, col_pred = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image, caption="Uploaded MRI", width="stretch")
        st.caption(
            f"**Format:** {uploaded_file.type} &nbsp;·&nbsp; "
            f"**Size:** {uploaded_file.size / 1024:.1f} KB &nbsp;·&nbsp; "
            f"**Dimensions:** {image.size[0]}×{image.size[1]} px"
        )

    with col_pred:
        # Prediction card
        badge_color, badge_icon = CLASS_BADGE_COLORS.get(
            pred_label, ("#1E88E5", "🔵")
        )
        st.markdown(
            f"""
            <div class="pred-card">
                <div style="font-size:0.9rem; color:#aaa; text-transform:uppercase; letter-spacing:2px; margin-bottom:0.6rem;">Prediction</div>
                <div class="pred-label" style="color:{badge_color};">{badge_icon} {display_name}</div>
                <div style="font-size:0.85rem; color:#aaa; text-transform:uppercase; letter-spacing:2px; margin:1rem 0 0.3rem 0;">Confidence</div>
                <div class="pred-conf">{confidence:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")  # spacer

        # Probability bars for every class
        st.markdown("##### Class Probabilities")
        for i, cls in enumerate(CLASS_NAMES):
            prob = float(probabilities[i]) * 100
            color, icon = CLASS_BADGE_COLORS.get(cls, ("#1E88E5", "🔵"))
            st.markdown(
                f"""
                <div class="prob-row">
                    <span>{icon} {DISPLAY_NAMES[cls]}</span>
                    <span style="font-weight:600;">{prob:.1f}%</span>
                </div>
                <div class="prob-bar-outer">
                    <div class="prob-bar-inner" style="width:{prob}%; background:{color};"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Grad-CAM Visualization ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 Explainable AI — Grad‑CAM++")
    st.caption(
        "The heatmap highlights the regions the model relied on for its decision."
    )

    col_hm, col_ov = st.columns(2)
    with col_hm:
        heatmap_rgb = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB,
        )
        st.image(heatmap_rgb, caption="Grad‑CAM++ Heatmap", width="stretch")
    with col_ov:
        st.image(
            overlay,
            caption=f"Overlay — {display_name} ({confidence:.2%})",
            width="stretch",
        )

    # ── Confidence indicator ──────────────────────────────────────────────
    st.markdown("---")
    if confidence > 0.9:
        st.success(
            f"✅ **High confidence** ({confidence:.2%}) — the model is very sure."
        )
    elif confidence > 0.7:
        st.info(
            f"ℹ️ **Moderate confidence** ({confidence:.2%}) — consider a medical review."
        )
    else:
        st.warning(
            f"❗ **Low confidence** ({confidence:.2%}) — a professional review is strongly advised."
        )

    # ── Medical Disclaimer ────────────────────────────────────────────────
    st.warning(
        "**⚕️ Medical Disclaimer:** This tool is for educational and research purposes only. "
        "It does **not** provide medical diagnoses. Always consult a qualified healthcare professional."
    )


# ── Page view: Model Performance ─────────────────────────────────────────────
def render_model_performance(summary):
    """Render model analytics, hyperparameters, and training visualizations."""

    if not summary:
        st.info("Project summary could not be loaded from `artifacts/logs/project_summary.json`.")
        return

    st.subheader("📊 Model Performance & Optimization Analytics")

    # ── Top-level metric cards ────────────────────────────────────────────
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Test Accuracy", f"{summary.get('final_test_accuracy', 0):.2%}")
    with col_m2:
        st.metric("Validation Loss", f"{summary.get('best_validation_loss', 0):.4f}")
    with col_m3:
        st.metric("Training Epochs", summary.get("final_training_epochs", 0))
    with col_m4:
        st.metric("Optuna Trials", summary.get("total_trials", 0))

    st.markdown("---")

    # ── Hyperparameters in two columns ────────────────────────────────────
    st.markdown("### ⚙️ Configuration (Optuna Optimized)")
    hp = summary.get("best_hyperparameters", {})

    col_train, col_model = st.columns(2, gap="large")

    with col_train:
        st.markdown("#### 🏋️ Training")
        train_df = pd.DataFrame(
            {
                "Parameter": [
                    "Backbone Learning Rate",
                    "Head Learning Rate",
                    "Weight Decay",
                    "Batch Size",
                    "Scheduler Patience",
                ],
                "Value": [
                    f"{hp.get('backbone_lr', 0):.2e}",
                    f"{hp.get('head_lr', 0):.2e}",
                    f"{hp.get('weight_decay', 0):.2e}",
                    str(hp.get("batch_size", "-")),
                    str(hp.get("scheduler_patience", "-")),
                ],
            }
        )
        st.dataframe(train_df, hide_index=True, width="stretch")

    with col_model:
        st.markdown("#### 🧠 Model")
        arch = summary.get("model", MODEL_NAME).replace("_", "-").upper()
        model_df = pd.DataFrame(
            {
                "Parameter": [
                    "Architecture",
                    "Unfrozen Blocks",
                    "Image Size",
                    "Explainability",
                    "Device",
                ],
                "Value": [
                    arch,
                    str(hp.get("unfreeze_blocks", "-")),
                    f"{IMAGE_SIZE}×{IMAGE_SIZE}",
                    "Grad-CAM++",
                    str(DEVICE).upper(),
                ],
            }
        )
        st.dataframe(model_df, hide_index=True, width="stretch")

    st.markdown("---")

    # ── Per-class metrics ─────────────────────────────────────────────────
    report = summary.get("classification_report", {})
    class_rows = []
    for cls in CLASS_NAMES:
        cr = report.get(cls, {})
        if cr:
            class_rows.append(
                {
                    "Class": DISPLAY_NAMES[cls],
                    "Precision": f"{cr.get('precision', 0):.2%}",
                    "Recall": f"{cr.get('recall', 0):.2%}",
                    "F1-Score": f"{cr.get('f1-score', 0):.2%}",
                    "Support": int(cr.get("support", 0)),
                }
            )
    if class_rows:
        st.markdown("### 📋 Per-Class Classification Report")
        st.dataframe(
            pd.DataFrame(class_rows), hide_index=True, width="stretch"
        )

    st.markdown("---")

    # ── Training Visualizations ───────────────────────────────────────────
    st.markdown("### 📈 Training Visualizations")

    learning_curve_path = os.path.abspath(
        os.path.join(web_dir, "..", "artifacts", "logs", "final_learning_curve.png")
    )
    confusion_matrix_path = os.path.abspath(
        os.path.join(
            web_dir, "..", "artifacts", "logs", "confusion_matrix_optuna_final.png"
        )
    )

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if os.path.exists(learning_curve_path):
            st.image(
                learning_curve_path,
                caption="Final Learning Curve (Loss / Accuracy)",
                width="stretch",
            )
        else:
            st.info("Learning curve image not found.")
    with col_img2:
        if os.path.exists(confusion_matrix_path):
            st.image(
                confusion_matrix_path,
                caption="Confusion Matrix — Test Dataset",
                width="stretch",
            )
        else:
            st.info("Confusion matrix image not found.")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Brain MRI Classifier",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Hero header ───────────────────────────────────────────────────────
    st.markdown(
        '<p class="hero-title">🧠 Brain MRI Tumor Classifier</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-subtitle">'
        "AI‑Assisted Brain MRI Classification<br>"
        "with Explainable Grad‑CAM++"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Load resources ────────────────────────────────────────────────────
    summary = get_project_summary()
    model, gradcam = load_cached_model()
    if model is None:
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("ℹ️ About")
        test_acc = summary.get("final_test_accuracy", 0) if summary else 0
        st.markdown(
            f"**EfficientNet‑B2** classifier for brain MRI scans.  \n"
            f"Test accuracy: **{test_acc:.2%}**"
        )

        st.markdown("#### Tumor Classes")
        for cls in CLASS_NAMES:
            color, icon = CLASS_BADGE_COLORS.get(cls, ("#1E88E5", "🔵"))
            label = DISPLAY_NAMES[cls]
            st.markdown(
                f'<span class="class-badge" style="background:{color}22; color:{color}; border:1px solid {color}55;">'
                f"{icon} {label}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.header("🎛️ Grad‑CAM Settings")
        heatmap_sensitivity = st.slider(
            "Heatmap sensitivity",
            min_value=0.05,
            max_value=0.60,
            value=0.25,
            step=0.05,
            help="Lower → looser tinting · Higher → tighter focus.",
        )
        heatmap_sharpness = st.slider(
            "Heatmap sharpness (γ)",
            min_value=1.0,
            max_value=3.0,
            value=1.8,
            step=0.2,
            help="Higher → suppresses weak activation, tightening highlights.",
        )

    # ── Navigation buttons ────────────────────────────────────────────────
    if "page" not in st.session_state:
        st.session_state.page = "analysis"

    nav_col1, nav_col2, _ = st.columns([1, 1, 2])
    with nav_col1:
        if st.button(
            "🔍 Scan Analysis",
            use_container_width=True,
            type="primary" if st.session_state.page == "analysis" else "secondary",
        ):
            st.session_state.page = "analysis"
            st.rerun()
    with nav_col2:
        if st.button(
            "📊 Model Performance & Analytics",
            use_container_width=True,
            type="primary" if st.session_state.page == "performance" else "secondary",
        ):
            st.session_state.page = "performance"
            st.rerun()

    st.markdown("---")

    # ── Render active page ────────────────────────────────────────────────
    if st.session_state.page == "analysis":
        render_scan_analysis(model, gradcam, heatmap_sensitivity, heatmap_sharpness)
    else:
        render_model_performance(summary)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div class="footer-wrapper">
            <div class="footer-title">🧠 Brain MRI Classifier</div>
            <div class="footer-tech">PyTorch &bull; EfficientNet‑B2 &bull; Grad‑CAM++ &bull; Streamlit</div>
            <div class="footer-author">Developed by Arup Das</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
