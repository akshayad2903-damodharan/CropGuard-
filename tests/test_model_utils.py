"""
Unit Tests for CropGuard Model Utilities
Tests cover image preprocessing, label loading, label parsing, mock model prediction,
and error handling edge cases.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from PIL import Image

import config
import model_utils


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_rgb_image():
    """Creates a temporary 300x400 RGB test image."""
    img_array = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_rgba_image():
    """Creates a temporary 100x100 RGBA test image with transparency."""
    img_array = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_grayscale_image():
    """Creates a temporary 150x150 Grayscale test image."""
    img_array = np.random.randint(0, 256, (150, 150), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_classes():
    """Standard 4-class list for testing."""
    return [
        "Apple___Apple_scab",
        "Apple___healthy",
        "Tomato___Early_blight",
        "Tomato___healthy"
    ]


class MockKerasModel:
    """Mock Keras model for testing inference without loading heavy binaries."""
    def __init__(self, output_probs=None, input_shape=(None, 224, 224, 3)):
        self.output_probs = output_probs
        self.input_shape = input_shape

    def predict(self, x, verbose=0):
        if self.output_probs is not None:
            return np.array([self.output_probs], dtype=np.float32)
        # Default 4-class output
        return np.array([[0.05, 0.85, 0.08, 0.02]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------

def test_preprocess_image_shape_and_channels(sample_rgb_image):
    """Test that preprocessing produces (1, 224, 224, 3) shape."""
    processed = model_utils.preprocess_image(sample_rgb_image, target_size=(224, 224))
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (1, 224, 224, 3)
    assert processed.dtype == np.float32


def test_preprocess_image_converts_rgba_to_3_channels(sample_rgba_image):
    """Test that 4-channel RGBA images are converted to 3-channel RGB."""
    processed = model_utils.preprocess_image(sample_rgba_image, target_size=(224, 224))
    assert processed.shape == (1, 224, 224, 3)


def test_preprocess_image_converts_grayscale_to_3_channels(sample_grayscale_image):
    """Test that 1-channel Grayscale images are converted to 3-channel RGB."""
    processed = model_utils.preprocess_image(sample_grayscale_image, target_size=(224, 224))
    assert processed.shape == (1, 224, 224, 3)


def test_preprocess_scaling_modes(sample_rgb_image):
    """Test all 3 supported preprocessing modes produce expected ranges."""
    # Mode 1: Scaled 0 to 1
    scaled = model_utils.preprocess_image(sample_rgb_image, mode="scaled_0_1")
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0

    # Mode 2: MobileNetV2 normalized -1 to 1
    mobilenet = model_utils.preprocess_image(sample_rgb_image, mode="mobilenet_v2")
    assert mobilenet.min() >= -1.0
    assert mobilenet.max() <= 1.0

    # Mode 3: Raw pixels 0 to 255
    raw = model_utils.preprocess_image(sample_rgb_image, mode="raw_0_255")
    assert raw.min() >= 0.0
    assert raw.max() <= 255.0


def test_preprocess_invalid_input():
    """Test that invalid types or dimensions raise appropriate exceptions."""
    with pytest.raises(TypeError):
        model_utils.preprocess_image(12345)

    with pytest.raises(ValueError):
        model_utils.preprocess_image(Image.new("RGB", (50, 50)), target_size=(0, -10))


# ---------------------------------------------------------------------------
# Label Parsing Tests
# ---------------------------------------------------------------------------

def test_parse_class_label_disease():
    """Test parsing disease class labels."""
    res = model_utils.parse_class_label("Tomato___Early_blight")
    assert res["crop"] == "Tomato"
    assert res["condition"] == "Early Blight"
    assert res["is_healthy"] is False


def test_parse_class_label_healthy():
    """Test parsing healthy class labels."""
    res = model_utils.parse_class_label("Apple___healthy")
    assert res["crop"] == "Apple"
    assert res["condition"] == "Healthy"
    assert res["is_healthy"] is True


def test_parse_class_label_complex_names():
    """Test parsing special crop names with parentheses and underscores."""
    res = model_utils.parse_class_label("Corn_(maize)___Common_rust_")
    assert "Corn" in res["crop"]
    assert "Common Rust" in res["condition"]
    assert res["is_healthy"] is False


def test_parse_class_label_edge_cases():
    """Test fallback when label is empty or None."""
    res = model_utils.parse_class_label("")
    assert res["crop"] == "Unknown Crop"
    assert res["is_healthy"] is False


# ---------------------------------------------------------------------------
# Class Label Loading Tests
# ---------------------------------------------------------------------------

def test_load_class_names_json_list():
    """Test loading class names from a JSON array."""
    classes = ["Tomato___healthy", "Tomato___Early_blight"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(classes, f)
        temp_path = f.name

    try:
        loaded = model_utils.load_class_names(temp_path)
        assert loaded == classes
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_load_class_names_json_dict():
    """Test loading class names from a JSON dictionary."""
    classes_dict = {"0": "Apple___healthy", "1": "Apple___Black_rot"}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(classes_dict, f)
        temp_path = f.name

    try:
        loaded = model_utils.load_class_names(temp_path)
        assert loaded == ["Apple___healthy", "Apple___Black_rot"]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_load_class_names_missing_file():
    """Test error handling when JSON file is missing."""
    with pytest.raises(FileNotFoundError):
        model_utils.load_class_names("non_existent_classes_file_xyz.json")


def test_load_class_names_malformed_json():
    """Test error handling on malformed JSON content."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write("{ invalid json format : [ ")
        temp_path = f.name

    try:
        with pytest.raises(ValueError):
            model_utils.load_class_names(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------------------------
# Prediction & Inference Tests
# ---------------------------------------------------------------------------

def test_predict_leaf_disease_success(sample_classes):
    """Test standard prediction pipeline with mock model."""
    mock_model = MockKerasModel(output_probs=[0.05, 0.85, 0.08, 0.02])
    fake_img_batch = np.zeros((1, 224, 224, 3), dtype=np.float32)

    result = model_utils.predict_leaf_disease(mock_model, fake_img_batch, sample_classes)

    assert result["predicted_class"] == "Apple___healthy"
    assert result["crop"] == "Apple"
    assert result["condition"] == "Healthy"
    assert result["is_healthy"] is True
    assert result["confidence_pct"] == 85.0
    assert len(result["top_predictions"]) == 4
    assert result["top_predictions"][0]["class_name"] == "Apple___healthy"


def test_predict_leaf_disease_logits_conversion(sample_classes):
    """Test that raw logits (unnormalized) are properly converted to softmax probabilities."""
    # Raw unnormalized logits
    raw_logits = [2.0, 5.0, 1.0, 0.0]
    mock_model = MockKerasModel(output_probs=raw_logits)
    fake_img_batch = np.zeros((1, 224, 224, 3), dtype=np.float32)

    result = model_utils.predict_leaf_disease(mock_model, fake_img_batch, sample_classes)
    # Highest logit index 1 should be predicted class
    assert result["predicted_class"] == "Apple___healthy"
    assert 0.0 <= result["confidence_score"] <= 1.0
    # Probabilities should sum to 1.0
    probs = result["raw_probabilities"]
    assert np.isclose(np.sum(probs), 1.0)


def test_predict_leaf_disease_score_mismatch_error(sample_classes):
    """Test that mismatched score count and class labels count raises ValueError."""
    # 3 outputs vs 4 classes in sample_classes
    mock_model = MockKerasModel(output_probs=[0.6, 0.3, 0.1])
    fake_img_batch = np.zeros((1, 224, 224, 3), dtype=np.float32)

    with pytest.raises(ValueError) as excinfo:
        model_utils.predict_leaf_disease(mock_model, fake_img_batch, sample_classes)
    assert "Class count mismatch" in str(excinfo.value)


def test_predict_leaf_disease_non_finite_scores(sample_classes):
    """Test that NaN or Inf model outputs raise ValueError."""
    mock_model = MockKerasModel(output_probs=[np.nan, 0.5, 0.3, 0.2])
    fake_img_batch = np.zeros((1, 224, 224, 3), dtype=np.float32)

    with pytest.raises(ValueError) as excinfo:
        model_utils.predict_leaf_disease(mock_model, fake_img_batch, sample_classes)
    assert "non-finite" in str(excinfo.value)


def test_get_model_input_shape():
    """Test extracting input dimensions from model."""
    mock_model = MockKerasModel(input_shape=(None, 256, 256, 3))
    shape = model_utils.get_model_input_shape(mock_model)
    assert shape == (256, 256)

    # Fallback test
    fallback = model_utils.get_model_input_shape(None)
    assert fallback == config.DEFAULT_INPUT_SHAPE
