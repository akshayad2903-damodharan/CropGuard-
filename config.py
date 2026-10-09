"""
CropGuard Configuration Module
Contains application settings, theme definitions, file paths, and default parameters.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# Base Workspace Directory
BASE_DIR = Path(__file__).resolve().parent

# Application Branding
APP_TITLE = "CropGuard"
APP_SUBTITLE = "Deep Learning-Based Crop Disease Classification"
APP_TAGLINE = "Smarter insights from crop leaf images."
APP_ICON = "🌿"
APP_VERSION = "1.0.0"

# Model and Label Paths
DEFAULT_MODEL_PATH = os.getenv("CROPGUARD_MODEL_PATH", str(BASE_DIR / "models" / "cropguard_model.keras"))
FALLBACK_MODEL_PATHS = [
    str(BASE_DIR / "models" / "cropguard_model.keras"),
    str(BASE_DIR / "cropguard_model.keras"),
    str(BASE_DIR / "models" / "cropguard_model.h5"),
    str(BASE_DIR / "cropguard_model.h5"),
    str(BASE_DIR / "model" / "cropguard_model.keras"),
]

DEFAULT_CLASS_NAMES_PATH = os.getenv("CROPGUARD_CLASSES_PATH", str(BASE_DIR / "models" / "class_names.json"))
FALLBACK_CLASS_NAMES_PATHS = [
    str(BASE_DIR / "models" / "class_names.json"),
    str(BASE_DIR / "class_names.json"),
    str(BASE_DIR / "model" / "class_names.json"),
]

# Model Inference Configuration
DEFAULT_INPUT_SHAPE: Tuple[int, int] = (224, 224)

# Preprocessing Modes:
# A. 'raw_0_255'      -> Raw RGB pixel values in [0, 255] float32
# B. 'scaled_0_1'    -> Pixel values normalized to [0, 1] (img / 255.0) [DEFAULT]
# C. 'mobilenet_v2'  -> MobileNetV2 preprocessing: normalized to [-1, 1] ((img / 127.5) - 1.0)
PREPROCESSING_MODES: Dict[str, str] = {
    "scaled_0_1": "Scaled (0.0 to 1.0) [Standard CNN / ResNet]",
    "mobilenet_v2": "MobileNetV2 Normalized (-1.0 to 1.0)",
    "raw_0_255": "Raw Pixels (0 to 255) [Internal Rescaling Layer]",
}
DEFAULT_PREPROCESSING_MODE = "scaled_0_1"

# Upload Constraints
ALLOWED_IMAGE_TYPES: List[str] = ["jpg", "jpeg", "png"]
MAX_IMAGE_SIZE_MB: int = 15

# UI Color Palette (Refined Enterprise Dark Agricultural Theme)
THEME_COLORS = {
    "primary": "#42B883",       # Primary Emerald Accent
    "secondary": "#276749",     # Secondary Forest Green
    "accent": "#42B883",        # Vibrant Emerald
    "highlight": "#52B788",     # Fresh Mint Highlight
    "light_mint": "rgba(66, 184, 131, 0.15)", # Subtle Mint Tint
    "background": "#0D1117",    # Deep Charcoal Application Background
    "sidebar": "#171F1B",       # Sidebar Background
    "surface": "#151D19",       # Card Surface
    "surface_card": "#1B2620",  # Elevated Card Surface
    "border": "rgba(166, 181, 170, 0.16)",       # Subtle Border
    "border_accent": "rgba(66, 184, 131, 0.24)", # Accent Border
    "text_dark": "#F1F5F2",     # Primary Text (Light Contrast)
    "text_muted": "#A6B5AA",    # Secondary Text (Muted Sage)
    "success": "#42B883",       # Emerald Green for Healthy
    "warning": "#F59E0B",       # Amber Warning
    "danger": "#EF4444",        # Crimson Red for Severe Disease
}

# PlantVillage Standard Reference Classes (38 Classes for reference / fallback template)
PLANT_VILLAGE_CLASSES: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Static Illustrative Mock Results for Explicit Demo Mode
# Note: These are static illustrative samples strictly used when Demo Mode is manually enabled
# or when no trained model is loaded, so testers can inspect the UI layout.
DEMO_STATIC_SAMPLE = {
    "predicted_class": "Tomato___Early_blight",
    "confidence_pct": 92.4,
    "top_predictions": [
        {"class_name": "Tomato___Early_blight", "crop": "Tomato", "condition": "Early blight", "is_healthy": False, "score": 0.924, "pct": 92.4},
        {"class_name": "Tomato___Late_blight", "crop": "Tomato", "condition": "Late blight", "is_healthy": False, "score": 0.051, "pct": 5.1},
        {"class_name": "Tomato___Target_Spot", "crop": "Tomato", "condition": "Target Spot", "is_healthy": False, "score": 0.015, "pct": 1.5},
        {"class_name": "Tomato___Septoria_leaf_spot", "crop": "Tomato", "condition": "Septoria leaf spot", "is_healthy": False, "score": 0.007, "pct": 0.7},
        {"class_name": "Tomato___healthy", "crop": "Tomato", "condition": "Healthy", "is_healthy": True, "score": 0.003, "pct": 0.3},
    ],
    "explanation": "Illustrative static example demonstrating confidence visualization and probability breakdown.",
}
