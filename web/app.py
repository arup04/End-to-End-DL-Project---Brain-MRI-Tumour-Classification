import os
import sys
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
    MODEL_PATH
)
from brain_mri.inference.predictor import (
    load_model_weights,
    preprocess_image,
    predict
)
from brain_mri.inference.gradcam import (
    GradCAM,
    get_target_layer,
    generate_heatmap,
    create_overlay
)

@st.cache_resource
def load_cached_model():
    """Load model and create a cached GradCAM instance"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found!")
        return None, None

    try:
        model = load_model_weights(MODEL_PATH, DEVICE, CLASS_NAMES)
        target_layer = get_target_layer(model)
        gradcam = GradCAM(model, target_layer)
        return model, gradcam
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def main():
    st.set_page_config(page_title="Brain MRI Classifier", page_icon="🧠", layout="wide",
                       initial_sidebar_state="expanded")

    # -------- Custom CSS ----------
    st.markdown("""
    <style>
        .main-header { font-size: 2.5rem; font-weight: bold; color: #1E88E5; text-align: center; margin-bottom: 0.5rem; }
        .sub-header  { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 1.5rem; }
        .footer { text-align: center; margin-top: 3rem; color: #888; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)

    # ---------- Header ----------
    st.markdown('<p class="main-header">🧠 Brain MRI Tumor Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI‑Assisted Brain MRI Classification with Explainable Grad‑CAM</p>', unsafe_allow_html=True)

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses **EfficientNet‑B2** to classify brain MRI scans into four categories:
        - Glioma
        - Meningioma
        - No Tumor
        - Pituitary Tumor

        **Model Performance:**
        - Training accuracy: 90.39%
        - Validation accuracy: 92.04%
        - Test accuracy: 90.01%

        **Technology:** PyTorch · Grad‑CAM · Streamlit
        """)
        st.header("📊 Class Information")
        st.info("""
        **Glioma** – most common primary brain tumor  
        **Meningioma** – tumor arising from meninges  
        **Pituitary Tumor** – tumor in pituitary gland  
        **No Tumor** – healthy brain scan
        """)

        st.header("🎛️ Grad‑CAM Settings")
        heatmap_sensitivity = st.slider(
            "Heatmap sensitivity",
            min_value=0.05, max_value=0.60, value=0.25, step=0.05,
            help="Lower = more of the scan gets tinted (looser). "
                 "Higher = only the strongest activation is shown (tighter)."
        )
        heatmap_sharpness = st.slider(
            "Heatmap sharpness (gamma)",
            min_value=1.0, max_value=3.0, value=1.8, step=0.2,
            help="Higher = suppresses weak/medium activation more, "
                 "tightening the highlighted region down to the strongest evidence."
        )

    # ---------- Load model once ----------
    model, gradcam = load_cached_model()
    if model is None:
        st.stop()

    # ---------- File upload ----------
    uploaded_file = st.file_uploader(
        "📤 Upload an MRI Scan",
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI image (JPG, JPEG, or PNG)"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📷 Uploaded MRI")
            image = Image.open(uploaded_file).convert('RGB')
            # Limit image width for cleaner layout
            st.image(image, width=500)
        with col2:
            st.subheader("📋 Image Details")
            st.markdown(f"""
            **Format:** {uploaded_file.type}  
            **Size:** {uploaded_file.size/1024:.1f} KB  
            **Dimensions:** {image.size[0]} × {image.size[1]} px
            """)

        if st.button("🔍 Analyze MRI Scan", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is analyzing the scan…"):
                try:
                    tensor = preprocess_image(image, IMAGE_SIZE, DEVICE)
                    pred_idx, confidence, probabilities = predict(model, tensor)
                    pred_label = CLASS_NAMES[pred_idx]
                    display_name = DISPLAY_NAMES[pred_label]

                    # Grad‑CAM
                    heatmap = generate_heatmap(gradcam, tensor, pred_idx, image, gamma=heatmap_sharpness)
                    overlay = create_overlay(image, heatmap, threshold=heatmap_sensitivity)

                    # ====== 1. Top Prediction Badge ======
                    st.markdown("---")
                    st.success(f"🟢 **{display_name}**  \n**{confidence:.2%}** Confidence")

                    # ====== 2. Classification Result ======
                    st.subheader("🧠 MRI Classification Result")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Predicted Class", display_name)
                    with col_b:
                        st.metric("Confidence", f"{confidence:.2%}")

                    # ====== 3. Grad‑CAM Visualization ======
                    st.markdown("---")
                    st.subheader("🔬 Explainable AI (Grad‑CAM)")
                    st.caption("The heatmap highlights areas the model relied on for its decision.")
                    col1, col2 = st.columns(2)
                    with col1:
                        heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET),
                                                   cv2.COLOR_BGR2RGB)
                        st.image(heatmap_rgb, caption="Grad‑CAM Heatmap", width='stretch')
                    with col2:
                        st.image(overlay, caption=f"Overlay: {display_name} ({confidence:.2%})",
                                 width='stretch')

                    # ====== 4. Class Probabilities (percentage) ======
                    st.markdown("---")
                    st.subheader("📊 Class Probabilities")
                    prob_df = pd.DataFrame({
                        "Class": [DISPLAY_NAMES[c] for c in CLASS_NAMES],
                        "Confidence (%)": probabilities * 100
                    }).set_index("Class")
                    st.bar_chart(prob_df)

                    # Simplified table
                    st.markdown("#### Detailed Breakdown")
                    table_df = pd.DataFrame({
                        "Class": [DISPLAY_NAMES[c] for c in CLASS_NAMES],
                        "Confidence": [f"{p*100:.2f}%" for p in probabilities]
                    })
                    st.dataframe(table_df, width='stretch', hide_index=True)

                    # ====== 5. Confidence indicator ======
                    st.markdown("---")
                    if confidence > 0.9:
                        st.success(f"✅ High confidence ({confidence:.2%}) – the model is very sure.")
                    elif confidence > 0.7:
                        st.info(f"ℹ️ Moderate confidence ({confidence:.2%}) – consider a medical review.")
                    else:
                        st.warning(f"❗ Low confidence ({confidence:.2%}) – a professional review is strongly advised.")

                    # ====== 6. Medical Disclaimer ======
                    st.warning("""
                    **⚕️ Medical Disclaimer:** This tool is for educational and research purposes only.  
                    It does **not** provide medical diagnoses. Always consult a qualified healthcare professional.
                    """)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
    else:
        st.info("📤 Upload an MRI scan above to get started!")
        st.markdown("""
        ### 📝 How to use
        1. Click **Browse files** and select a brain MRI image.
        2. Press **Analyze MRI Scan**.
        3. View the prediction and Grad‑CAM heatmap.

        **Supported formats:** JPG, JPEG, PNG  
        **Recommended size:** 224×224 pixels or larger
        """)

    # ---------- Footer ----------
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        Built with 
        <b>PyTorch</b> · <b>EfficientNet‑B2</b> · <b>Grad‑CAM</b> · <b>Streamlit</b><br>
        Developed by Arup Das
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
