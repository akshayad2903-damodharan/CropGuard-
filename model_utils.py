"""
CropGuard Model Utility Module
Provides robust, cached functions for loading TensorFlow/Keras models,
loading class label mappings, preprocessing leaf images, and executing inference.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import streamlit as st

import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cropguard")


# ---------------------------------------------------------------------------
# Label Parsing Utilities
# ---------------------------------------------------------------------------

def parse_class_label(raw_label: str) -> Dict[str, Union[str, bool]]:
    """
    Parses a raw class label into readable Crop and Condition strings.
    Correctly detects healthy leaves versus disease states.

    Examples:
        - "Tomato___Early_blight" -> Crop: "Tomato", Condition: "Early Blight", Healthy: False
        - "Apple___healthy" -> Crop: "Apple", Condition: "Healthy", Healthy: True
        - "Corn_(maize)___Common_rust_" -> Crop: "Corn (maize)", Condition: "Common Rust", Healthy: False
        - "healthy_potato" -> Crop: "Potato", Condition: "Healthy", Healthy: True
    """
    if not raw_label or not isinstance(raw_label, str):
        return {
            "crop": "Unknown Crop",
            "condition": "Unspecified Condition",
            "is_healthy": False,
            "raw_label": str(raw_label) if raw_label is not None else "",
        }

    raw = raw_label.strip()
    is_healthy = bool(re.search(r"(?:^|[\W_])healthy(?:[\W_]|$)", raw, re.IGNORECASE))

    # PlantVillage style: "Crop___Disease_Name"
    if "___" in raw:
        parts = raw.split("___", 1)
        crop_part = parts[0].replace("_", " ").strip()
        disease_part = parts[1].replace("_", " ").strip()
    elif "__" in raw:
        parts = raw.split("__", 1)
        crop_part = parts[0].replace("_", " ").strip()
        disease_part = parts[1].replace("_", " ").strip()
    elif "-" in raw and not raw.startswith("-"):
        parts = raw.split("-", 1)
        crop_part = parts[0].replace("_", " ").strip()
        disease_part = parts[1].replace("_", " ").strip()
    else:
        # Fallback split
        crop_part = "Plant"
        disease_part = raw.replace("_", " ").strip()

    # Clean up formatting
    crop = crop_part.title() if crop_part.islower() else crop_part
    
    if is_healthy:
        condition = "Healthy"
    else:
        # Capitalize words nicely while preserving special terms
        words = disease_part.split()
        clean_words = [w.capitalize() if w.islower() else w for w in words]
        condition = " ".join(clean_words) if clean_words else "Disease Detected"

    return {
        "crop": crop,
        "condition": condition,
        "is_healthy": is_healthy,
        "raw_label": raw,
    }


# ---------------------------------------------------------------------------
# File Resolution Helpers
# ---------------------------------------------------------------------------

def resolve_model_path(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Locates an existing Keras/TensorFlow model file from custom path or standard locations.
    """
    if custom_path:
        p = Path(custom_path)
        return str(p.resolve()) if p.exists() and p.is_file() else None

    for candidate_str in config.FALLBACK_MODEL_PATHS:
        candidate = Path(candidate_str)
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    return None


def resolve_class_names_path(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Locates an existing class names JSON file from custom path or standard locations.
    """
    if custom_path:
        p = Path(custom_path)
        return str(p.resolve()) if p.exists() and p.is_file() else None

    for candidate_str in config.FALLBACK_CLASS_NAMES_PATHS:
        candidate = Path(candidate_str)
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    return None


# ---------------------------------------------------------------------------
# Model & Class Label Loading (Cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str):
    """
    Loads a trained TensorFlow/Keras model from the specified file path.
    Cached via Streamlit to ensure fast inference without reloading on reruns.
    """
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "TensorFlow is not installed in the current Python environment. "
            "Please run: pip install tensorflow"
        ) from e

    try:
        # Load model using tf.keras
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
        raise RuntimeError(
            f"Failed to load model from '{model_path}'. "
            f"Ensure it is a valid Keras (.keras or .h5) model. Error: {str(e)}"
        ) from e


@st.cache_data(show_spinner=False)
def load_class_names(class_file_path: Optional[str] = None) -> List[str]:
    """
    Loads class names from a JSON file.
    Accepts JSON arrays: ["Apple___healthy", ...]
    or JSON dictionaries: {"0": "Apple___healthy", "1": ...}
    """
    resolved_path = resolve_class_names_path(class_file_path)

    if not resolved_path:
        # If no file exists, return empty or raise informative error
        raise FileNotFoundError(
            "Class names JSON file was not found. Please place `class_names.json` "
            "in the `models/` directory or project root."
        )

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            class_names = [str(item) for item in data]
        elif isinstance(data, dict):
            # Sort by numerical key if possible, else alphabetical
            try:
                sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
                class_names = [str(v) for _, v in sorted_items]
            except ValueError:
                class_names = [str(v) for _, v in data.items()]
        else:
            raise ValueError(f"Invalid JSON format in {resolved_path}. Expected array or key-value object.")

        if not class_names:
            raise ValueError(f"Class names file '{resolved_path}' contains an empty list.")

        logger.info(f"Loaded {len(class_names)} class names from {resolved_path}")
        return class_names
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON syntax in '{resolved_path}': {e}") from e
    except Exception as e:
        raise RuntimeError(f"Could not load class names from '{resolved_path}': {e}") from e


def get_model_input_shape(model: Any) -> Tuple[int, int]:
    """
    Inspects a loaded Keras model to extract expected input (height, width).
    Falls back to config.DEFAULT_INPUT_SHAPE (224, 224) if dynamic or undefined.
    """
    fallback = config.DEFAULT_INPUT_SHAPE
    if model is None:
        return fallback

    try:
        # Check model.input_shape or model.inputs
        input_shape = getattr(model, "input_shape", None)
        if input_shape is None and hasattr(model, "inputs") and model.inputs:
            input_shape = model.inputs[0].shape

        if input_shape:
            # Typically (batch, height, width, channels)
            if len(input_shape) == 4:
                h = input_shape[1]
                w = input_shape[2]
                if h is not None and w is not None and int(h) > 0 and int(w) > 0:
                    return (int(h), int(w))
    except Exception as e:
        logger.warning(f"Could not automatically determine model input shape: {e}. Using fallback {fallback}.")

    return fallback


# ---------------------------------------------------------------------------
# Image Preprocessing Pipeline
# ---------------------------------------------------------------------------

def preprocess_image(
    image: Union[Image.Image, str, Path, bytes],
    target_size: Tuple[int, int] = config.DEFAULT_INPUT_SHAPE,
    mode: str = config.DEFAULT_PREPROCESSING_MODE,
) -> np.ndarray:
    """
    Dedicated image preprocessing pipeline.
    1. Opens & validates image safely.
    2. Converts to standard RGB (3 channels).
    3. Resizes to target (height, width).
    4. Converts to float32 numpy array.
    5. Applies exactly the selected normalization mode:
       - 'scaled_0_1': [0, 1] range (division by 255.0)
       - 'mobilenet_v2': [-1, 1] range ((x / 127.5) - 1.0)
       - 'raw_0_255': [0, 255] float32 range
    6. Adds batch dimension -> (1, height, width, 3).
    """
    # 1. Open safely
    if isinstance(image, (str, Path)):
        pil_img = Image.open(image)
    elif isinstance(image, bytes):
        import io
        pil_img = Image.open(io.BytesIO(image))
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}. Expected PIL.Image, str, Path, or bytes.")

    # 2. RGB conversion (strips alpha channel if PNG, converts grayscale to 3 channels)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # 3. Resize safely
    target_h, target_w = target_size
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target dimensions: {target_size}. Dimensions must be positive.")

    # Note: PIL resize takes (width, height)
    resized_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # 4. Convert to NumPy float32 array
    img_array = np.array(resized_img, dtype=np.float32)

    # 5. Apply exactly the configured preprocessing mode
    if mode == "scaled_0_1":
        processed_array = img_array / 255.0
    elif mode == "mobilenet_v2":
        # Matches tf.keras.applications.mobilenet_v2.preprocess_input
        processed_array = (img_array / 127.5) - 1.0
    elif mode == "raw_0_255":
        processed_array = img_array
    else:
        logger.warning(f"Unknown preprocessing mode '{mode}'. Falling back to 'scaled_0_1'.")
        processed_array = img_array / 255.0

    # 6. Add batch dimension -> (1, height, width, 3)
    batch_array = np.expand_dims(processed_array, axis=0)
    return batch_array


# ---------------------------------------------------------------------------
# Prediction & Inference Pipeline
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax implementation for 1D arrays."""
    shift_x = x - np.max(x)
    exps = np.exp(shift_x)
    return exps / np.sum(exps)


def predict_leaf_disease(
    model: Any,
    preprocessed_img: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Runs model inference on preprocessed image batch and returns structured prediction results.
    Validates output dimensions, finiteness, and class count matching.
    """
    if model is None:
        raise ValueError("Model instance is None. Cannot run inference.")

    if not class_names:
        raise ValueError("Class names list is empty. Cannot map model outputs.")

    # Validate input batch shape
    if not isinstance(preprocessed_img, np.ndarray):
        raise TypeError(f"Expected preprocessed_img as numpy ndarray, got {type(preprocessed_img)}")

    if len(preprocessed_img.shape) != 4 or preprocessed_img.shape[0] != 1 or preprocessed_img.shape[3] != 3:
        raise ValueError(
            f"Expected input batch shape (1, H, W, 3), but received shape {preprocessed_img.shape}"
        )

    # Run inference
    try:
        raw_output = model.predict(preprocessed_img, verbose=0)
    except Exception as e:
        logger.error(f"Inference error during model.predict: {e}", exc_info=True)
        raise RuntimeError(f"Error during model prediction: {str(e)}") from e

    # Convert to standard numpy array
    if hasattr(raw_output, "numpy"):
        raw_output = raw_output.numpy()
    raw_output = np.array(raw_output)

    # Validate output shape
    if raw_output.size == 0:
        raise ValueError("Model returned an empty prediction array.")

    # Flatten batch dimension if present
    if raw_output.ndim == 2 and raw_output.shape[0] == 1:
        scores = raw_output[0]
    elif raw_output.ndim == 1:
        scores = raw_output
    else:
        raise ValueError(f"Unexpected model output shape: {raw_output.shape}")

    # Check for NaN or Inf
    if not np.all(np.isfinite(scores)):
        raise ValueError("Model produced non-finite (NaN or Inf) output scores.")

    # Validate that score count matches class names count
    num_scores = len(scores)
    num_classes = len(class_names)
    if num_scores != num_classes:
        raise ValueError(
            f"Class count mismatch: The loaded model outputs {num_scores} score(s), "
            f"but class_names.json defines {num_classes} class label(s). "
            "Please ensure the class labels match the trained model's output layer exactly."
        )

    # Verify if outputs are probabilities or raw logits
    # If any score is negative or the sum deviates significantly from 1.0, apply softmax
    score_sum = float(np.sum(scores))
    has_negative = bool(np.any(scores < 0))
    if has_negative or abs(score_sum - 1.0) > 0.05:
        probabilities = softmax(scores)
    else:
        # Normalize in case of minor floating point rounding
        probabilities = scores / score_sum if score_sum > 0 else softmax(scores)

    # Predicted class
    pred_idx = int(np.argmax(probabilities))
    predicted_raw_label = class_names[pred_idx]
    confidence_score = float(probabilities[pred_idx])
    confidence_pct = round(confidence_score * 100.0, 1)

    # Parse crop and condition
    parsed_info = parse_class_label(predicted_raw_label)

    # Build ranked predictions table
    ranked_indices = np.argsort(probabilities)[::-1]
    top_predictions = []
    for idx in ranked_indices:
        label = class_names[idx]
        parsed = parse_class_label(label)
        score = float(probabilities[idx])
        top_predictions.append({
            "class_name": label,
            "crop": parsed["crop"],
            "condition": parsed["condition"],
            "is_healthy": parsed["is_healthy"],
            "score": score,
            "pct": round(score * 100.0, 1),
        })

    return {
        "predicted_class": predicted_raw_label,
        "crop": parsed_info["crop"],
        "condition": parsed_info["condition"],
        "is_healthy": parsed_info["is_healthy"],
        "confidence_score": confidence_score,
        "confidence_pct": confidence_pct,
        "top_predictions": top_predictions,
        "raw_probabilities": probabilities,
        "is_demo": False,
    }


# ---------------------------------------------------------------------------
# System Status Inspection
# ---------------------------------------------------------------------------

def inspect_system_status() -> Dict[str, Any]:
    """
    Inspects availability of model, labels, and TensorFlow without throwing unhandled exceptions.
    Used by the Streamlit UI to render setup diagnostics and switch to Demo mode if needed.
    """
    model_path = resolve_model_path()
    classes_path = resolve_class_names_path()

    tf_available = False
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        tf_available = True
    except ImportError:
        tf_version = "Not Installed"

    model_loaded = False
    model_error = None
    input_shape = config.DEFAULT_INPUT_SHAPE
    num_classes = 0

    if model_path and tf_available:
        try:
            model = load_keras_model(model_path)
            model_loaded = True
            input_shape = get_model_input_shape(model)
        except Exception as e:
            model_error = str(e)

    classes_loaded = False
    class_names = []
    classes_error = None

    if classes_path:
        try:
            class_names = load_class_names(classes_path)
            classes_loaded = True
            num_classes = len(class_names)
        except Exception as e:
            classes_error = str(e)

    return {
        "tf_available": tf_available,
        "tf_version": tf_version,
        "model_path": model_path,
        "model_loaded": model_loaded,
        "model_error": model_error,
        "classes_path": classes_path,
        "classes_loaded": classes_loaded,
        "classes_error": classes_error,
        "input_shape": input_shape,
        "num_classes": num_classes,
        "is_ready": (model_loaded and classes_loaded and model_error is None and classes_error is None),
    }
