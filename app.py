"""
CropGuard — AI-Powered Crop Disease Classification
Main Streamlit Application Dashboard
Refined Enterprise AI Precision Agriculture Interface
"""

import io
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

import config
import model_utils

# Try importing Plotly for interactive visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ---------------------------------------------------------------------------
# Page Configuration & Global Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=f"{config.APP_TITLE} — {config.APP_SUBTITLE}",
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_custom_css():
    """Injects refined enterprise dark-mode agricultural AI dashboard CSS."""
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

            :root {{
                --primary: {config.THEME_COLORS['primary']};
                --secondary: {config.THEME_COLORS['secondary']};
                --accent: {config.THEME_COLORS['accent']};
                --highlight: {config.THEME_COLORS['highlight']};
                --light-mint: {config.THEME_COLORS['light_mint']};
                --bg: {config.THEME_COLORS['background']};
                --sidebar-bg: {config.THEME_COLORS.get('sidebar', '#171F1B')};
                --surface: {config.THEME_COLORS['surface']};
                --surface-card: {config.THEME_COLORS['surface_card']};
                --border: {config.THEME_COLORS['border']};
                --border-accent: {config.THEME_COLORS.get('border_accent', 'rgba(66, 184, 131, 0.24)')};
                --text-primary: {config.THEME_COLORS['text_dark']};
                --text-secondary: {config.THEME_COLORS['text_muted']};
                --success: {config.THEME_COLORS['success']};
                --danger: {config.THEME_COLORS['danger']};
                --warning: {config.THEME_COLORS['warning']};
            }}

            /* Base App Background & Global Typography */
            .stApp, [data-testid="stAppViewContainer"] {{
                background-color: var(--bg) !important;
                color: var(--text-primary) !important;
                font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            }}

            [data-testid="stHeader"] {{
                background: transparent !important;
            }}

            /* Sidebar Theme */
            [data-testid="stSidebar"] {{
                background-color: var(--sidebar-bg) !important;
                border-right: 1px solid var(--border) !important;
            }}

            [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{
                color: var(--text-secondary);
            }}

            [data-testid="stSidebar"] h3 {{
                color: var(--text-primary) !important;
                font-family: 'Outfit', sans-serif !important;
                font-size: 1.1rem !important;
                font-weight: 600 !important;
                letter-spacing: -0.01em;
            }}

            /* Custom Radio Tab Navigation in Sidebar */
            [data-testid="stSidebar"] [data-testid="stRadio"] > div {{
                gap: 8px;
            }}

            [data-testid="stSidebar"] [data-testid="stRadio"] label {{
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 0.65rem 0.95rem;
                transition: all 0.2s ease-in-out;
                color: var(--text-secondary) !important;
                cursor: pointer;
                width: 100%;
                display: flex;
                align-items: center;
            }}

            [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
                background: var(--surface-card);
                border-color: var(--border-accent);
                color: var(--text-primary) !important;
            }}

            [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
            [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {{
                background: linear-gradient(135deg, rgba(39, 103, 73, 0.4) 0%, rgba(66, 184, 131, 0.2) 100%) !important;
                border-color: var(--primary) !important;
                color: var(--primary) !important;
                font-weight: 600 !important;
                box-shadow: 0 2px 10px rgba(66, 184, 131, 0.15);
            }}

            /* Typography */
            h1, h2, h3, h4, .brand-title {{
                font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif !important;
                font-weight: 700;
                color: var(--text-primary) !important;
                letter-spacing: -0.02em;
            }}

            p, span, div {{
                color: var(--text-secondary);
            }}

            strong, b {{
                color: var(--text-primary);
            }}

            /* Header Branding Hero Banner */
            .header-container {{
                background: linear-gradient(135deg, #0F231A 0%, #173627 50%, #1F4C37 100%);
                padding: 2.2rem 2.6rem;
                border-radius: 18px;
                color: white;
                margin-bottom: 2rem;
                border: 1px solid rgba(66, 184, 131, 0.28);
                box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35);
                display: flex;
                align-items: center;
                justify-content: space-between;
                position: relative;
                overflow: hidden;
            }}

            .header-container::after {{
                content: "🌿";
                position: absolute;
                right: -25px;
                bottom: -35px;
                font-size: 11.5rem;
                opacity: 0.05;
                pointer-events: none;
            }}

            .header-badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                background: rgba(66, 184, 131, 0.16);
                backdrop-filter: blur(8px);
                padding: 0.35rem 0.85rem;
                border-radius: 9999px;
                font-size: 0.78rem;
                font-weight: 600;
                color: #42B883;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.65rem;
                border: 1px solid rgba(66, 184, 131, 0.35);
            }}

            .header-title {{
                font-size: 2.35rem;
                font-weight: 800;
                margin: 0;
                line-height: 1.15;
                color: #FFFFFF !important;
                letter-spacing: -0.03em;
            }}

            .header-subtitle {{
                font-size: 1.08rem;
                color: #C3D9C9;
                margin-top: 0.35rem;
                font-weight: 500;
            }}

            .header-tagline {{
                font-size: 0.92rem;
                color: #42B883;
                margin-top: 0.3rem;
                font-style: italic;
                font-weight: 500;
            }}

            /* Cohesive Dark Card System */
            .custom-card {{
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 1.5rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
                transition: all 0.2s ease-in-out;
            }}

            .custom-card:hover {{
                border-color: var(--border-accent);
                box-shadow: 0 6px 22px rgba(0, 0, 0, 0.3);
            }}

            .card-title {{
                font-size: 1.12rem;
                font-weight: 700;
                color: var(--text-primary);
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 0.65rem;
                font-family: 'Outfit', sans-serif !important;
            }}

            /* Diagnostic Result Cards */
            .result-healthy-card {{
                background: linear-gradient(135deg, rgba(39, 103, 73, 0.35) 0%, rgba(21, 29, 25, 0.95) 100%);
                border: 1px solid rgba(66, 184, 131, 0.45);
                border-radius: 14px;
                padding: 1.6rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 8px 24px rgba(66, 184, 131, 0.12);
            }}

            .result-disease-card {{
                background: linear-gradient(135deg, rgba(220, 38, 38, 0.22) 0%, rgba(21, 29, 25, 0.95) 100%);
                border: 1px solid rgba(239, 68, 68, 0.45);
                border-radius: 14px;
                padding: 1.6rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 8px 24px rgba(220, 38, 38, 0.12);
            }}

            .demo-banner {{
                background: rgba(245, 158, 11, 0.12);
                border: 1px solid rgba(245, 158, 11, 0.35);
                border-radius: 12px;
                padding: 1rem 1.25rem;
                color: #FBBF24;
                font-weight: 500;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin-bottom: 1.25rem;
            }}

            /* Status Pills */
            .status-pill {{
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.25rem 0.65rem;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }}

            .pill-active {{
                background: rgba(66, 184, 131, 0.16);
                color: #42B883;
                border: 1px solid rgba(66, 184, 131, 0.35);
            }}

            .pill-warning {{
                background: rgba(245, 158, 11, 0.16);
                color: #FBBF24;
                border: 1px solid rgba(245, 158, 11, 0.35);
            }}

            .pill-danger {{
                background: rgba(239, 68, 68, 0.16);
                color: #F87171;
                border: 1px solid rgba(239, 68, 68, 0.35);
            }}

            /* Step Card Flow for How It Works */
            .step-card {{
                background: var(--surface);
                border: 1px solid var(--border);
                border-left: 4px solid var(--primary);
                border-radius: 12px;
                padding: 1.25rem 1.4rem;
                margin-bottom: 1rem;
                box-shadow: 0 3px 12px rgba(0, 0, 0, 0.18);
                transition: all 0.2s ease;
            }}

            .step-card:hover {{
                border-color: var(--border-accent);
                border-left: 4px solid var(--highlight);
                background: var(--surface-card);
            }}

            .step-number {{
                background: var(--light-mint);
                color: var(--primary);
                font-weight: 800;
                font-size: 0.8rem;
                padding: 0.2rem 0.6rem;
                border-radius: 6px;
                display: inline-block;
                margin-bottom: 0.4rem;
                border: 1px solid var(--border-accent);
            }}

            /* Metadata Chips */
            .meta-badge {{
                background: var(--surface-card);
                color: var(--text-secondary);
                padding: 0.35rem 0.7rem;
                border-radius: 8px;
                font-size: 0.82rem;
                font-weight: 500;
                display: inline-block;
                margin-right: 0.4rem;
                margin-bottom: 0.4rem;
                border: 1px solid var(--border);
            }}

            /* Streamlit Primary Button Custom Styling */
            div.stButton > button:first-child {{
                background: linear-gradient(135deg, #276749 0%, #42B883 100%) !important;
                color: #FFFFFF !important;
                font-weight: 600 !important;
                font-size: 0.98rem !important;
                padding: 0.65rem 1.8rem !important;
                border-radius: 10px !important;
                border: none !important;
                box-shadow: 0 4px 16px rgba(66, 184, 131, 0.25) !important;
                transition: all 0.2s ease !important;
            }}

            div.stButton > button:first-child:hover {{
                background: linear-gradient(135deg, #2D7A56 0%, #4CD497 100%) !important;
                box-shadow: 0 6px 24px rgba(66, 184, 131, 0.4) !important;
                transform: translateY(-1px) !important;
                color: #FFFFFF !important;
            }}

            div.stButton > button:first-child:active {{
                transform: translateY(0px) !important;
            }}

            /* File Uploader Section Styling */
            [data-testid="stFileUploader"] section {{
                background: var(--surface) !important;
                border: 1px dashed var(--border) !important;
                border-radius: 12px !important;
                padding: 1.2rem !important;
                transition: all 0.2s ease !important;
            }}

            [data-testid="stFileUploader"] section:hover {{
                border-color: var(--primary) !important;
                background: var(--surface-card) !important;
            }}

            /* Streamlit Native Selectbox & Input Container Styling */
            [data-testid="stSelectbox"] > div > div {{
                background-color: var(--surface) !important;
                border: 1px solid var(--border) !important;
                border-radius: 8px !important;
                color: var(--text-primary) !important;
            }}

            /* Progress Bar Accent */
            [data-testid="stProgress"] > div > div > div > div {{
                background: linear-gradient(90deg, #276749 0%, #42B883 100%) !important;
                border-radius: 6px !important;
            }}

            /* Expanders */
            [data-testid="stExpander"] {{
                background: var(--surface) !important;
                border: 1px solid var(--border) !important;
                border-radius: 12px !important;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
                margin-bottom: 0.75rem !important;
            }}

            [data-testid="stExpander"] summary {{
                color: var(--text-primary) !important;
                font-weight: 600 !important;
            }}

            /* Horizontal Divider */
            hr {{
                border-color: var(--border) !important;
                opacity: 0.6 !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

def init_session_state():
    """Initializes app session state variables."""
    if "active_nav" not in st.session_state:
        st.session_state.active_nav = "Disease Prediction"
    if "force_demo_mode" not in st.session_state:
        st.session_state.force_demo_mode = False
    if "preprocessing_mode" not in st.session_state:
        st.session_state.preprocessing_mode = config.DEFAULT_PREPROCESSING_MODE
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "uploaded_image_bytes" not in st.session_state:
        st.session_state.uploaded_image_bytes = None
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None


# ---------------------------------------------------------------------------
# Header Component
# ---------------------------------------------------------------------------

def render_header():
    """Renders the top branded hero banner."""
    st.markdown(
        f"""
        <div class="header-container">
            <div>
                <div class="header-badge">🌿 AI Precision Agriculture • V{config.APP_VERSION}</div>
                <h1 class="header-title">{config.APP_TITLE}</h1>
                <div class="header-subtitle">{config.APP_SUBTITLE}</div>
                <div class="header-tagline">"{config.APP_TAGLINE}"</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar Navigation & Settings Panel
# ---------------------------------------------------------------------------

def render_sidebar(system_status: Dict[str, Any]):
    """Renders the sidebar navigation and model configuration controls."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 0.75rem 0 1.25rem 0;">
                <div style="font-size: 2.4rem; margin-bottom: 0.25rem;">🌿</div>
                <h3 style="margin: 0; color: #42B883 !important; font-size: 1.35rem; font-weight: 700;">{config.APP_TITLE}</h3>
                <p style="margin: 0; font-size: 0.8rem; color: #A6B5AA;">Precision Agriculture Diagnostics</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🧭 Navigation")
        selected_page = st.radio(
            label="Select Page",
            options=["Disease Prediction", "How It Works", "About the Project"],
            index=["Disease Prediction", "How It Works", "About the Project"].index(st.session_state.active_nav),
            key="nav_radio",
            label_visibility="collapsed",
        )
        st.session_state.active_nav = selected_page

        st.divider()

        # Model & Pipeline Configuration
        st.markdown("### ⚙️ Inference Pipeline")

        # Status badge in sidebar
        if system_status["is_ready"]:
            st.markdown(
                '<span class="status-pill pill-active">● Real Model Loaded</span>',
                unsafe_allow_html=True,
            )
            model_subtext = f"Trained classes: **{system_status['num_classes']}**"
        else:
            st.markdown(
                '<span class="status-pill pill-warning">▲ Model Setup Pending</span>',
                unsafe_allow_html=True,
            )
            model_subtext = "Place model in `models/` folder"

        st.caption(model_subtext)

        # Preprocessing selector
        selected_mode = st.selectbox(
            "Preprocessing Setting",
            options=list(config.PREPROCESSING_MODES.keys()),
            format_func=lambda x: config.PREPROCESSING_MODES[x],
            index=list(config.PREPROCESSING_MODES.keys()).index(st.session_state.preprocessing_mode),
            help="Choose the normalization method matching the Google Colab training notebook.",
        )
        st.session_state.preprocessing_mode = selected_mode

        # Demo mode toggle
        st.markdown("---")
        st.markdown("### 🧪 Demonstration Controls")
        demo_toggled = st.checkbox(
            "Force Demo Mode (Placeholder)",
            value=st.session_state.force_demo_mode,
            help="Simulates the interface using static illustrative results without invoking model inference.",
        )
        st.session_state.force_demo_mode = demo_toggled

        if not system_status["is_ready"] and not demo_toggled:
            st.info("💡 Real model not detected. Demo mode will automatically provide static illustrative preview.")

        st.divider()
        st.markdown(
            """
            <div style="font-size: 0.75rem; color: #6E8274; text-align: center;">
                Academic Prototype for Deep Learning Coursework.<br>
                Always verify on-field crop diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page 1: Disease Prediction Dashboard
# ---------------------------------------------------------------------------

def render_disease_prediction_page(system_status: Dict[str, Any]):
    """Renders the main upload, image preview, analysis, and results dashboard."""
    st.markdown("## 🔍 Crop Disease Diagnostics")
    st.markdown(
        "Upload a high-resolution photograph of an individual crop leaf to classify plant condition "
        "and obtain confidence breakdown."
    )

    # Determine whether we are in Real Model Mode or Demo Mode
    is_real_inference = system_status["is_ready"] and not st.session_state.force_demo_mode

    # Setup alert if model is not loaded
    if not system_status["is_ready"]:
        st.markdown(
            f"""
            <div class="demo-banner">
                <div style="font-size: 1.4rem;">⚠️</div>
                <div>
                    <strong style="color: #FBBF24;">Trained Model Awaiting Integration</strong><br>
                    <span style="font-size: 0.88rem; color: #E2E8F0;">
                        No trained model found at <code>models/cropguard_model.keras</code>.
                        Add your exported Keras model and <code>class_names.json</code> into the <code>models/</code> folder.
                        Meanwhile, you can explore the dashboard using <strong>Demo Mode (Static Illustrative Preview)</strong> below.
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Layout Columns: Left (Upload & Preview), Right (Analysis & Results)
    col_upload, col_results = st.columns([1.05, 1.15], gap="large")

    with col_upload:
        st.markdown(
            """
            <div class="custom-card">
                <div class="card-title">📤 1. Upload Leaf Image</div>
                <p style="font-size: 0.88rem; color: #A6B5AA; margin-bottom: 0.8rem;">
                    Accepted formats: <strong>JPG, JPEG, PNG</strong> (Max 15MB).<br>
                    <em>Tip: Ensure the leaf is centered, well-illuminated, and clearly visible.</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload leaf photograph",
            type=config.ALLOWED_IMAGE_TYPES,
            help="Select a single clear crop leaf photo.",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            st.session_state.uploaded_image_bytes = image_bytes
            st.session_state.uploaded_filename = uploaded_file.name

            try:
                pil_image = Image.open(io.BytesIO(image_bytes))
                img_width, img_height = pil_image.size
                file_size_kb = len(image_bytes) / 1024.0

                st.markdown(
                    """
                    <div class="custom-card" style="padding: 1rem;">
                        <div class="card-title" style="font-size: 1rem; margin-bottom: 0.5rem;">🖼️ Image Preview & Metadata</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.image(pil_image, caption=f"Preview: {uploaded_file.name}", use_container_width=True)

                # Metadata chips
                st.markdown(
                    f"""
                    <div style="margin-top: 0.5rem;">
                        <span class="meta-badge">📁 {uploaded_file.name}</span>
                        <span class="meta-badge">📐 {img_width} × {img_height} px</span>
                        <span class="meta-badge">🎨 Mode: {pil_image.mode}</span>
                        <span class="meta-badge">💾 {file_size_kb:.1f} KB</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                
                # Primary action button
                if st.button("🌿 Analyze Leaf", use_container_width=True, type="primary"):
                    with st.spinner("Processing image through neural pipeline..."):
                        time.sleep(0.35)  # Smooth UX transition
                        if is_real_inference:
                            # Real inference using trained model
                            try:
                                model = model_utils.load_keras_model(system_status["model_path"])
                                class_names = model_utils.load_class_names(system_status["classes_path"])
                                target_shape = system_status["input_shape"]
                                
                                preprocessed_batch = model_utils.preprocess_image(
                                    pil_image,
                                    target_size=target_shape,
                                    mode=st.session_state.preprocessing_mode,
                                )

                                start_t = time.time()
                                pred_result = model_utils.predict_leaf_disease(
                                    model=model,
                                    preprocessed_img=preprocessed_batch,
                                    class_names=class_names,
                                )
                                latency_ms = (time.time() - start_t) * 1000.0
                                pred_result["latency_ms"] = latency_ms
                                pred_result["target_shape"] = target_shape
                                pred_result["preprocessing_mode"] = st.session_state.preprocessing_mode
                                pred_result["filename"] = uploaded_file.name
                                pred_result["is_demo"] = False
                                st.session_state.analysis_result = pred_result
                            except Exception as e:
                                st.error(f"Inference Error: {str(e)}")
                                st.session_state.analysis_result = None
                        else:
                            # Demo Mode (Static illustrative results)
                            static_demo = config.DEMO_STATIC_SAMPLE.copy()
                            parsed_demo = model_utils.parse_class_label(static_demo["predicted_class"])
                            st.session_state.analysis_result = {
                                "predicted_class": static_demo["predicted_class"],
                                "crop": parsed_demo["crop"],
                                "condition": parsed_demo["condition"],
                                "is_healthy": parsed_demo["is_healthy"],
                                "confidence_score": static_demo["confidence_pct"] / 100.0,
                                "confidence_pct": static_demo["confidence_pct"],
                                "top_predictions": static_demo["top_predictions"],
                                "latency_ms": 14.2,
                                "target_shape": config.DEFAULT_INPUT_SHAPE,
                                "preprocessing_mode": st.session_state.preprocessing_mode,
                                "filename": uploaded_file.name,
                                "is_demo": True,
                            }

            except Exception as e:
                st.error(f"Unable to read image file: {e}")
        else:
            # Empty state
            st.markdown(
                """
                <div class="custom-card" style="text-align: center; padding: 2.8rem 1.5rem; background: #151D19; border-style: dashed; border-width: 1px;">
                    <div style="font-size: 3.2rem; margin-bottom: 0.5rem; opacity: 0.7;">🍃</div>
                    <h4 style="color: #42B883; margin-bottom: 0.35rem;">No Image Uploaded Yet</h4>
                    <p style="font-size: 0.86rem; color: #A6B5AA; max-width: 320px; margin: 0 auto;">
                        Select or drag-and-drop a leaf photograph above to view image preview and begin classification.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_results:
        st.markdown(
            """
            <div class="custom-card">
                <div class="card-title">📊 2. Diagnostic Results & Insights</div>
                <p style="font-size: 0.88rem; color: #A6B5AA; margin-bottom: 0.2rem;">
                    Detailed classification predictions, confidence metrics, and probability distribution.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        result = st.session_state.analysis_result

        if result is not None:
            # Banner if Demo Mode
            if result.get("is_demo", False):
                st.markdown(
                    """
                    <div class="demo-banner" style="margin-bottom: 1rem;">
                        <div style="font-size: 1.2rem;">ℹ️</div>
                        <div style="font-size: 0.85rem;">
                            <strong style="color: #FBBF24;">DEMO MODE — PLACEHOLDER RESULTS</strong><br>
                            <span style="color: #E2E8F0;">This is a static illustrative example. Real model inference will automatically run once your trained Keras model is loaded.</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Highlighting Healthy vs Disease
            is_healthy = result.get("is_healthy", False)
            crop_name = result.get("crop", "Unknown")
            condition_name = result.get("condition", "Unknown")
            conf_pct = result.get("confidence_pct", 0.0)
            raw_label = result.get("predicted_class", "N/A")

            if is_healthy:
                st.markdown(
                    f"""
                    <div class="result-healthy-card">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span class="status-pill pill-active">✓ Healthy Condition</span>
                            <span style="font-size: 0.85rem; color: #42B883; font-weight: 600;">Confidence: {conf_pct:.1f}%</span>
                        </div>
                        <h2 style="color: #42B883 !important; margin: 0.6rem 0 0.2rem 0; font-size: 1.7rem;">
                            {crop_name} — Healthy Leaf
                        </h2>
                        <p style="margin: 0; color: #A6B5AA; font-size: 0.92rem;">
                            No visible fungal, bacterial, or viral lesion patterns detected matching the classifier dictionary.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-disease-card">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span class="status-pill pill-danger">▲ Condition Detected</span>
                            <span style="font-size: 0.85rem; color: #F87171; font-weight: 600;">Confidence: {conf_pct:.1f}%</span>
                        </div>
                        <h2 style="color: #F87171 !important; margin: 0.6rem 0 0.2rem 0; font-size: 1.7rem;">
                            {crop_name} — {condition_name}
                        </h2>
                        <p style="margin: 0; color: #E2E8F0; font-size: 0.92rem;">
                            Symptoms characteristic of <strong style="color: #FCA5A5;">{condition_name}</strong> observed on the uploaded leaf image.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Confidence Metric & Progress Gauge
            st.markdown("#### Confidence Breakdown")
            st.progress(min(max(float(conf_pct / 100.0), 0.0), 1.0))
            st.caption(f"Top predicted class confidence: **{conf_pct:.1f}%** | Raw Label: `{raw_label}`")

            # Probability Distribution Chart
            top_preds = result.get("top_predictions", [])
            if top_preds:
                # Take top 5 for visual clarity
                display_preds = top_preds[:5]
                df_preds = pd.DataFrame(display_preds)
                # Display clean name
                df_preds["Display_Name"] = df_preds.apply(
                    lambda row: f"{row['crop']} - {row['condition']}", axis=1
                )

                if HAS_PLOTLY:
                    # Invert order for top-down horizontal bar chart
                    df_plot = df_preds.iloc[::-1]
                    fig = px.bar(
                        df_plot,
                        x="pct",
                        y="Display_Name",
                        orientation="h",
                        text="pct",
                        labels={"pct": "Probability (%)", "Display_Name": "Predicted Category"},
                        color="pct",
                        color_continuous_scale=["#276749", "#34A872", "#42B883"],
                    )
                    fig.update_traces(
                        texttemplate="%{text:.1f}%",
                        textposition="outside",
                        marker_line_color="#42B883",
                        marker_line_width=1,
                    )
                    fig.update_layout(
                        margin=dict(l=10, r=40, t=10, b=10),
                        height=240,
                        coloraxis_showscale=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#F1F5F2", family="Plus Jakarta Sans"),
                        xaxis=dict(
                            range=[0, 105],
                            showgrid=True,
                            gridcolor="rgba(166, 181, 170, 0.12)",
                            tickfont=dict(color="#A6B5AA"),
                        ),
                        yaxis=dict(
                            title=None,
                            tickfont=dict(color="#F1F5F2"),
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Streamlit native bar chart fallback
                    st.bar_chart(
                        df_preds.set_index("Display_Name")["pct"],
                        horizontal=True,
                    )

            # Technical Summary & Pipeline Details
            with st.expander("🛠️ Inference Metadata & Diagnostics", expanded=False):
                st.markdown(
                    f"""
                    - **Model Input Dimensions**: `{result.get('target_shape', config.DEFAULT_INPUT_SHAPE)}`
                    - **Active Preprocessing**: `{config.PREPROCESSING_MODES.get(result.get('preprocessing_mode'), result.get('preprocessing_mode'))}`
                    - **Inference Latency**: `{result.get('latency_ms', 0.0):.1f} ms`
                    - **Evaluation Disclaimer**: Model outputs are mathematical classifications. Always cross-reference with professional agronomic guidance.
                    """
                )

            # Button to clear analysis
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Analyze Another Image", use_container_width=True):
                st.session_state.analysis_result = None
                st.session_state.uploaded_image_bytes = None
                st.rerun()

        else:
            # Empty state for results
            st.markdown(
                """
                <div class="custom-card" style="text-align: center; padding: 3rem 1.5rem; background: #151D19;">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem; opacity: 0.6;">📈</div>
                    <h4 style="color: #42B883; margin-bottom: 0.35rem;">Awaiting Analysis</h4>
                    <p style="font-size: 0.86rem; color: #A6B5AA; max-width: 320px; margin: 0 auto;">
                        Upload a leaf image on the left panel and click <strong>Analyze Leaf</strong> to compute class probabilities.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Page 2: How It Works
# ---------------------------------------------------------------------------

def render_how_it_works_page():
    """Renders the interactive educational pipeline explanation page."""
    st.markdown("## 🔬 How CropGuard Works")
    st.markdown(
        "CropGuard uses a deep convolutional neural network (CNN) trained on thousands of annotated leaf photos. "
        "Here is a step-by-step walkthrough of the computational pipeline from raw pixel input to diagnosis."
    )

    # Visual Stepper Pipeline
    steps = [
        {
            "step": "Step 1",
            "title": "Leaf Image Acquisition",
            "icon": "📸",
            "desc": "A digital photograph of a crop leaf is captured. For reliable inference, the leaf should be in sharp focus, well-lit, and isolated against a neutral background.",
        },
        {
            "step": "Step 2",
            "title": "Image Preprocessing & Resizing",
            "icon": "⚙️",
            "desc": "The raw image is converted to standard 3-channel RGB, resized to the neural network's expected resolution (e.g., 224×224 pixels), and pixel values are normalized (e.g. [0, 1] or MobileNet scaling) to ensure consistent numeric distribution.",
        },
        {
            "step": "Step 3",
            "title": "Convolutional Feature Extraction",
            "icon": "🧠",
            "desc": "Successive convolutional filters scan the image. Lower layers detect elementary edges and color gradients; intermediate layers identify leaf veins and textures; deeper layers detect complex disease lesion spots, chlorosis halos, and necrotic rings.",
        },
        {
            "step": "Step 4",
            "title": "Transfer Learning Backbone",
            "icon": "🚀",
            "desc": "Rather than training from scratch, transfer learning leverages feature extractors pretrained on large computer vision benchmarks (e.g., ImageNet, MobileNetV2, ResNet). Fine-tuning on agricultural datasets significantly reduces training time and prevents overfitting.",
        },
        {
            "step": "Step 5",
            "title": "Softmax Classification Head",
            "icon": "📊",
            "desc": "Dense classification layers map the extracted high-level features to output logits. A softmax activation transforms logits into a calibrated probability distribution summing to 1.0 across all target classes.",
        },
        {
            "step": "Step 6",
            "title": "Prediction & Confidence Output",
            "icon": "🎯",
            "desc": "The class with the highest probability (argmax) is parsed into the specific Crop species and Condition (Healthy or specific disease) alongside a percentage confidence score.",
        },
    ]

    for item in steps:
        st.markdown(
            f"""
            <div class="step-card">
                <span class="step-number">{item['step']}</span>
                <h4 style="margin: 0.2rem 0 0.4rem 0; color: #F1F5F2 !important; display: flex; align-items: center; gap: 0.4rem;">
                    <span>{item['icon']}</span> {item['title']}
                </h4>
                <p style="margin: 0; font-size: 0.9rem; color: #A6B5AA; line-height: 1.5;">
                    {item['desc']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📚 Core Deep Learning Concepts")

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        with st.expander("❓ What does a Convolutional Neural Network (CNN) do?", expanded=True):
            st.markdown(
                """
                A **Convolutional Neural Network** uses learnable spatial filters (kernels) that slide across the image matrix.
                Unlike traditional neural networks that treat pixels independently, CNNs preserve the 2D spatial arrangement of pixels,
                enabling the model to detect local visual patterns regardless of where they appear on the leaf.
                """
            )

        with st.expander("⚡ Why Transfer Learning saves training time & data", expanded=True):
            st.markdown(
                """
                Training deep vision models from scratch requires millions of labeled images and days of GPU compute.
                **Transfer learning** repurposes foundational feature representations (shapes, boundaries, textures)
                learned on massive datasets, requiring only a fraction of agricultural images to achieve high diagnostic accuracy.
                """
            )

    with col2:
        with st.expander("📏 Why Image Resizing & Normalization Matter", expanded=True):
            st.markdown(
                """
                Neural networks require fixed-dimension tensor inputs. Resizing enforces standard dimensions (e.g. 224×224).
                **Normalization** maps pixel integers (0–255) to bounded ranges (such as [0, 1] or [-1, 1]), keeping internal gradients
                stable and accelerating gradient descent convergence.
                """
            )

        with st.expander("🌾 Real-World Field Challenges & Validation", expanded=True):
            st.markdown(
                """
                Models trained on lab-collected datasets (e.g. PlantVillage) may encounter difficulty with:
                - Direct sunlight reflections and harsh shadows.
                - Soil, fingers, or cluttered agricultural background.
                - Multiple overlapping diseases or nutritional deficiencies on a single leaf.
                Rigorous validation on diverse field images is necessary before real-world deployment.
                """
            )


# ---------------------------------------------------------------------------
# Page 3: About the Project
# ---------------------------------------------------------------------------

def render_about_page(system_status: Dict[str, Any]):
    """Renders project specifications, architecture details, and academic disclaimer."""
    st.markdown("## ℹ️ About CropGuard")
    st.markdown(
        "**CropGuard** is an automated deep learning prototype developed for plant pathology diagnosis "
        "and precision agriculture assistance."
    )

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown(
            """
            <div class="custom-card">
                <div class="card-title">📋 Project Specifications</div>
                <table style="width: 100%; font-size: 0.9rem; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid rgba(166, 181, 170, 0.12);">
                        <td style="padding: 0.65rem 0; font-weight: 600; color: #42B883;">Project Name</td>
                        <td style="padding: 0.65rem 0; color: #F1F5F2;">CropGuard</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(166, 181, 170, 0.12);">
                        <td style="padding: 0.65rem 0; font-weight: 600; color: #42B883;">Category</td>
                        <td style="padding: 0.65rem 0; color: #A6B5AA;">Deep Learning / Computer Vision / Agriculture</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(166, 181, 170, 0.12);">
                        <td style="padding: 0.65rem 0; font-weight: 600; color: #42B883;">Frameworks</td>
                        <td style="padding: 0.65rem 0; color: #A6B5AA;">Python 3.12, TensorFlow / Keras, Streamlit, Pillow, NumPy, Plotly</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(166, 181, 170, 0.12);">
                        <td style="padding: 0.65rem 0; font-weight: 600; color: #42B883;">Reference Dataset</td>
                        <td style="padding: 0.65rem 0; color: #A6B5AA;">PlantVillage Dataset (54k+ labeled leaf images across 38 classes)</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.65rem 0; font-weight: 600; color: #42B883;">Architecture Type</td>
                        <td style="padding: 0.65rem 0; color: #A6B5AA;">Convolutional Transfer Learning (MobileNet / ResNet)</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="custom-card">
                <div class="card-title">🌱 Future Enhancements & Scope</div>
                <ul style="font-size: 0.9rem; color: #A6B5AA; padding-left: 1.2rem; line-height: 1.6;">
                    <li><strong style="color: #F1F5F2;">Severity Estimation</strong>: Segmenting necrotic lesion area to compute disease severity grade.</li>
                    <li><strong style="color: #F1F5F2;">Multi-Disease Co-Infection</strong>: Transitioning from single-label softmax to multi-label sigmoid classification.</li>
                    <li><strong style="color: #F1F5F2;">Offline Mobile Edge Deployment</strong>: Quantizing weights into TensorFlow Lite (TFLite) for low-power smartphones in rural areas.</li>
                    <li><strong style="color: #F1F5F2;">Agronomic Treatment Integration</strong>: Connecting diagnoses with validated organic and chemical remediation guidelines.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # Diagnostic & System Status Card
        st.markdown(
            """
            <div class="custom-card">
                <div class="card-title">🖥️ Runtime Environment & Status</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            - **TensorFlow Status**: `{'Installed (' + system_status['tf_version'] + ')' if system_status['tf_available'] else 'Not Found'}`
            - **Model File**: `{system_status['model_path'] if system_status['model_path'] else 'Not Detected (Waiting in models/)'}`
            - **Model Loading Status**: `{'✓ Loaded Successfully' if system_status['model_loaded'] else 'Pending'}`
            - **Class Labels File**: `{system_status['classes_path'] if system_status['classes_path'] else 'Not Detected'}`
            - **Configured Classes**: `{system_status['num_classes']} Classes`
            - **Input Resolution**: `{system_status['input_shape']}`
            """
        )

        st.markdown(
            """
            <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 12px; padding: 1rem; color: #FBBF24; font-size: 0.84rem; margin-top: 1rem;">
                <strong>⚠️ Educational Prototype Disclaimer</strong><br>
                <span style="color: #E2E8F0;">
                    CropGuard is built for college academic demonstration and educational exploration.
                    Predictions generated by this machine learning model are computational estimates and must not replace professional agronomic inspection, laboratory culture tests, or certified phytosanitary advice.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main Application Controller
# ---------------------------------------------------------------------------

def main():
    """Main application entry point."""
    init_session_state()
    inject_custom_css()
    render_header()

    # Inspect system readiness (model presence, labels, TF version)
    system_status = model_utils.inspect_system_status()

    # Render Sidebar
    render_sidebar(system_status)

    # Page Routing
    if st.session_state.active_nav == "Disease Prediction":
        render_disease_prediction_page(system_status)
    elif st.session_state.active_nav == "How It Works":
        render_how_it_works_page()
    elif st.session_state.active_nav == "About the Project":
        render_about_page(system_status)


if __name__ == "__main__":
    main()
