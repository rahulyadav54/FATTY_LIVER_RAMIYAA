"""
🏥 FATTY LIVER CLASSIFICATION - PREMIUM MEDICAL AI DASHBOARD
Deep Learning Ultrasound Analysis using Siamese ResNet-50
"""
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import os
import time

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from models.siamese_net import SiameseNetwork
from src.data_loader import get_data_transforms
from skimage import exposure

# Configure Streamlit page layout and metadata
st.set_page_config(
    page_title="Fatty Liver AI Clinical Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Medical CSS Styling
st.markdown("""
<style>
    /* Global Page Styling */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Custom Header Bar */
    .med-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 20px 28px;
        border-radius: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
        margin-bottom: 24px;
    }
    
    .med-header-title {
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        color: #ffffff;
    }

    .med-header-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
        font-weight: 400;
    }

    .med-badge {
        background: rgba(13, 148, 136, 0.2);
        color: #2dd4bf;
        border: 1px solid rgba(45, 212, 191, 0.3);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .med-badge-dot {
        width: 8px;
        height: 8px;
        background-color: #2dd4bf;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #2dd4bf;
    }

    /* Card Panels */
    .med-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
        margin-bottom: 20px;
    }

    .med-card-title {
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 10px;
    }

    /* Metric Cards */
    .stat-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .stat-label {
        font-size: 12px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stat-value {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 4px;
    }

    /* Status Indicators */
    .status-normal { color: #059669; background: #ecfdf5; border: 1px solid #a7f3d0; }
    .status-abnormal { color: #dc2626; background: #fef2f2; border: 1px solid #fecaca; }

    /* Class Cards in Sidebar */
    .class-info-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 10px;
    }
    
    .class-info-title {
        font-size: 13px;
        font-weight: 700;
        color: #0f172a;
    }

    .class-info-desc {
        font-size: 11px;
        color: #64748b;
        margin-top: 2px;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background: #fffbeb;
        border: 1px solid #fef3c7;
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 12px;
        color: #92400e;
        line-height: 1.5;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Optimized Model Loader (Cached once per session)
@st.cache_resource(show_spinner="Initializing Siamese ResNet-50 Network...")
def load_model():
    """Load model once into memory and cache resource"""
    t0 = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork().to(device)
    
    model_path = 'best_model.pth'
    if not Path(model_path).exists():
        st.error(f"❌ Model checkpoint file not found: {model_path}")
        return None, device, None, 0.0
    
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    
    model.eval()
    transform = get_data_transforms(is_train=False)
    load_time_ms = (time.perf_counter() - t0) * 1000
    
    return model, device, transform, load_time_ms


def is_valid_ultrasound(img_pil):
    """Validate if uploaded file has characteristics of an ultrasound image"""
    img_array = np.array(img_pil.convert('L'))
    
    if img_array.shape[0] < 100 or img_array.shape[1] < 100:
        return False, "Image resolution too small (< 100x100)"
    
    if img_array.shape[0] > 3000 or img_array.shape[1] > 3000:
        return False, "Image dimension exceeds maximum supported limit (3000x3000)"
    
    img_eq = exposure.equalize_adapthist(img_array / 255.0)
    contrast = img_eq.std()
    
    if contrast < 0.02:
        return False, "Low contrast image detected (Unlikely to be B-mode ultrasound)"
    
    if hasattr(img_pil, 'convert'):
        img_rgb = np.array(img_pil.convert('RGB'))
        color_std = img_rgb.std(axis=2).mean()
        if color_std > 50:
            return False, "High color variance detected (B-mode ultrasound is grayscale)"
    
    return True, "Valid B-mode Ultrasound"


def predict(image, model, device, transform, mode='multiclass'):
    """Fast, deterministic inference using torch.inference_mode()"""
    t0 = time.perf_counter()
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        probs_np = probs.cpu().numpy()[0]
        pred = int(np.argmax(probs_np))
        confidence = float(np.max(probs_np))
    
    latency_ms = (time.perf_counter() - t0) * 1000

    if mode == 'binary':
        binary_pred = 0 if pred == 0 else 1
        normal_prob = float(probs_np[0])
        abnormal_prob = float(np.sum(probs_np[1:]))
        binary_conf = normal_prob if binary_pred == 0 else abnormal_prob
        return binary_pred, np.array([normal_prob, abnormal_prob]), binary_conf, latency_ms
    
    return pred, probs_np, confidence, latency_ms


def main():
    # Top Medical Header Bar
    st.markdown("""
    <div class="med-header">
        <div>
            <div class="med-header-title">🏥 Fatty Liver AI Clinical Dashboard</div>
            <div class="med-header-subtitle">Siamese Neural Network (ResNet-50) B-mode Ultrasound Diagnostic Support</div>
        </div>
        <div class="med-badge">
            <span class="med-badge-dot"></span> System Online & Model Loaded
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load cached model
    model, device, transform, load_time_ms = load_model()
    
    if model is None:
        st.error("⚠️ Model failed to load. Please verify `best_model.pth` is in the working directory.")
        return

    # Sidebar Panel
    st.sidebar.markdown("### ⚙️ Analysis Configuration")
    classification_mode = st.sidebar.radio(
        "Classification Mode",
        ["Multi-class (5 Grades)", "Binary (Normal / Abnormal)"],
        help="Choose detailed 5-stage grading or high-level diagnostic triage"
    )
    mode = 'multiclass' if classification_mode == "Multi-class (5 Grades)" else 'binary'

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Supported Clinical Categories")

    class_info = [
        ("Normal (Grade 0)", "Healthy liver parenchyma with no steatosis (< 5% fat)", "#059669"),
        ("Grade-I (Mild)", "Mild steatosis with slight accumulation of fat (5 - 35% fat)", "#0891b2"),
        ("Grade-II (Moderate)", "Moderate steatosis with vessel blurring (35 - 65% fat)", "#d97706"),
        ("Grade-III (Severe)", "Severe steatosis with deep sound attenuation (> 65% fat)", "#dc2626"),
        ("CLD (Chronic)", "Advanced Chronic Liver Disease / Cirrhosis parenchymal scarring", "#7c3aed")
    ]

    for title, desc, color in class_info:
        st.sidebar.markdown(f"""
        <div class="class-info-card" style="border-left-color: {color};">
            <div class="class-info-title">{title}</div>
            <div class="class-info-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("🔬 Research Prototype | Version 2.0 | SNN-ResNet50")

    # Main Navigation Tabs
    tab_dashboard, tab_predict, tab_about, tab_guide = st.tabs([
        "📊 Dashboard", 
        "📤 Predict & Analyze", 
        "🧠 About Model", 
        "📖 User Guide"
    ])

    # Tab 1: Dashboard Overview
    with tab_dashboard:
        st.markdown("### 👋 Welcome to Ultrasound Liver Analysis")
        st.markdown("Use this clinical dashboard to upload B-mode liver ultrasound images, run Siamese neural network inference, and view probability distributions across steatosis severity grades.")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown("""
            <div class="stat-box">
                <div class="stat-label">Model Architecture</div>
                <div class="stat-value" style="font-size: 20px; color: #0284c7;">Siamese ResNet-50</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Model Warmup Latency</div>
                <div class="stat-value" style="font-size: 20px; color: #0d9488;">{load_time_ms:.1f} ms</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Compute Device</div>
                <div class="stat-value" style="font-size: 20px; color: #6366f1;">{str(device).upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown("""
            <div class="stat-box">
                <div class="stat-label">Evaluation Accuracy</div>
                <div class="stat-value" style="font-size: 20px; color: #059669;">Validated</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.info("💡 **Quick Start**: Navigate to the **Predict & Analyze** tab above to upload a liver ultrasound image and view real-time predictions.")

    # Tab 2: Prediction Workflow
    with tab_predict:
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("""
            <div class="med-card">
                <div class="med-card-title">📤 Upload Ultrasound Image</div>
            """, unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Select a B-mode ultrasound image file",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Supported formats: JPG, JPEG, PNG, BMP"
            )

            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption=f"Uploaded File: {uploaded_file.name}", use_container_width=True)
                
                is_valid, validation_msg = is_valid_ultrasound(image)
                if is_valid:
                    st.success(f"✓ Quality Check: {validation_msg}")
                else:
                    st.warning(f"⚠️ Image Validation Notice: {validation_msg}")
            
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown("""
            <div class="med-card">
                <div class="med-card-title">🔍 Diagnostic Inference & Probabilities</div>
            """, unsafe_allow_html=True)

            if uploaded_file:
                # Fast prediction execution
                pred_idx, probs, confidence, latency_ms = predict(image, model, device, transform, mode)

                if mode == 'binary':
                    class_names = ['Normal', 'Abnormal']
                    pred_label = class_names[pred_idx]
                    threshold = 0.60  # Validation-derived binary threshold (60%)
                else:
                    class_names = ['Normal', 'Grade-I', 'Grade-II', 'Grade-III', 'CLD']
                    pred_label = class_names[pred_idx]
                    threshold = 0.40  # Validation-derived 5-class threshold (40%)

                is_low_confidence = confidence < threshold

                if is_low_confidence:
                    st.markdown(f"""
                    <div style="padding: 16px; border-radius: 12px; margin-bottom: 16px; background: #fff7ed; border: 1px solid #fed7aa; color: #c2410c;">
                        <div style="font-size: 12px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; color: #ea580c;">⚠️ Uncertainty Alert</div>
                        <div style="font-size: 22px; font-weight: 800; margin-top: 4px; color: #9a3412;">Low-confidence prediction — further review required</div>
                        <div style="font-size: 13px; margin-top: 6px; color: #9a3412;">
                            Top candidate: <strong>{pred_label}</strong> (Confidence: <strong>{confidence * 100:.2f}%</strong> — below reliability threshold of {threshold * 100:.0f}%).
                            The model is uncertain. Result must not be used as a reliable classification.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    status_class = "status-normal" if (mode == 'binary' and pred_idx == 0) or (mode == 'multiclass' and pred_idx == 0) else "status-abnormal"
                    st.markdown(f"""
                    <div style="padding: 16px; border-radius: 12px; margin-bottom: 16px;" class="{status_class}">
                        <div style="font-size: 12px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Predicted Category</div>
                        <div style="font-size: 32px; font-weight: 800; margin-top: 2px;">{pred_label}</div>
                        <div style="font-size: 13px; margin-top: 4px;">Model Confidence: <strong>{confidence * 100:.2f}%</strong> (Inference time: {latency_ms:.1f} ms)</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("#### 📊 Category Probability Breakdown")
                prob_dict = {class_names[i]: float(probs[i] * 100) for i in range(len(class_names))}
                st.bar_chart(prob_dict)

                st.markdown("#### 📋 Detailed Numerical Breakdown")
                for cname, pval in prob_dict.items():
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.progress(min(1.0, max(0.0, pval / 100.0)))
                    with c2:
                        st.text(f"{pval:.2f}%")

                st.markdown("""
                <div class="disclaimer-box">
                    🔬 <strong>RESEARCH PROTOTYPE ONLY — NOT A DIAGNOSTIC TOOL</strong><br>
                    This application is an experimental research prototype for technical evaluation. It does not provide medical advice or definitive diagnostic findings. All outputs must be evaluated by qualified clinical professionals.
                </div>
                """, unsafe_allow_html=True)

            else:
                st.info("👆 Please upload an ultrasound image on the left panel to execute inference.")

            st.markdown("</div>", unsafe_allow_html=True)

    # Tab 3: About Model
    with tab_about:
        st.markdown("""
        ### 🧠 Architecture & Methodology Overview
        
        #### 1. Siamese Neural Network (SNN) Backbone
        - **Feature Extractor**: Modified ResNet-50 deep convolutional network pre-trained on ImageNet.
        - **Projection Head**: 2048 → 512 → 128 dimensional contrastive projection layer.
        - **Classifier Head**: Dense linear classification head producing 5 logits (`Normal`, `Grade-I`, `Grade-II`, `Grade-III`, `CLD`).

        #### 2. Self-Supervised Contrastive Pre-Training
        - Employs **NT-Xent (Normalized Temperature-scaled Cross Entropy Loss)** to bring representations of similar ultrasound tissue textures closer in feature space while pushing dissimilar grades apart.

        #### 3. Dataset Characteristics
        - Benchmark dataset: **IJCARS B-mode Steatosis Dataset** comprising 55 patient acquisitions (550 total B-mode ultrasound frames).
        - Multi-class target taxonomy:
          1. **Normal**: Steatosis < 5%
          2. **Grade-I (Mild)**: Steatosis 5% – 35%
          3. **Grade-II (Moderate)**: Steatosis 35% – 65%
          4. **Grade-III (Severe)**: Steatosis > 65%
          5. **CLD**: Chronic Liver Disease / Advanced Cirrhotic Changes
        """)

    # Tab 4: User Guide
    with tab_guide:
        st.markdown("""
        ### 📖 Operating Instructions & Troubleshooting

        #### How to Analyze an Ultrasound Image:
        1. Click on the **Predict & Analyze** tab.
        2. Drag and drop a B-mode ultrasound image into the upload area or click **Browse files**.
        3. Review the automatic quality validation message.
        4. Select **Multi-class (5 Grades)** or **Binary (Normal / Abnormal)** mode from the left sidebar depending on your analytical needs.
        5. Examine the predicted category, confidence percentage, and horizontal probability breakdown.

        #### Supported Image Requirements:
        - **Formats**: PNG, JPG, JPEG, BMP
        - **Modality**: Grayscale B-mode Liver Ultrasound
        - **Min Dimensions**: 100 x 100 pixels
        """)


if __name__ == '__main__':
    main()
