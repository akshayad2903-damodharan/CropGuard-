# 🌿 CropGuard — AI-Powered Crop Disease Classification

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-2D6A4F?logo=python)](https://python.org)
[![Framework](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Deep Learning](https://img.shields.io/badge/TensorFlow%2FKeras-2.15+-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **CropGuard** is a full-stack deep learning web application designed for automated crop pathology classification. Developed for precision agriculture research and academic demonstrations, it enables farmers, agronomists, and researchers to classify crop leaf images, view diagnostic confidence distributions, and inspect deep learning decision pipelines in real-time.

---

## 🌟 Key Features

* **Instant Leaf Disease Classification**: Upload leaf photos (`JPG`, `JPEG`, `PNG`) to detect fungal, bacterial, viral infections, or healthy leaf states.
* **Confidence & Probability Distribution**: Interactive horizontal bar charts showing top-ranked class probabilities and percentage metrics.
* **Seamless Teammate Model Integration**: Direct drop-in support for exported TensorFlow/Keras models (`.keras` or `.h5`) without requiring code changes.
* **Configurable Preprocessing Engine**:
  * Standard Scale `[0.0, 1.0]` (Default)
  * MobileNetV2 Normalization `[-1.0, 1.0]`
  * Raw Pixels `[0, 255]`
* **Dynamic Fallback & Safe Demo Mode**: Keeps the application interactive and presentable even when the model is still being trained or exported.
* **Educational Architecture Walkthrough**: Interactive visual pipeline breaking down CNN feature extraction, transfer learning, normalization, and validation.
* **Modern Forest-Green UI/UX**: Professional agricultural theme with card layouts, responsive previews, and accessible visual hierarchy.

---

## 📁 Project Directory Structure

```text
CropGuard/
├── app.py                     # Main Streamlit web application & UI
├── model_utils.py             # Model loading, caching, preprocessing, & inference engine
├── config.py                  # Global settings, paths, color tokens, & constants
├── requirements.txt           # Production & development dependencies
├── README.md                  # Comprehensive project documentation
├── .gitignore                 # Git ignore rules for Python, models, & caches
├── assets/                    # Static images, icons, and test leaf photos
│   └── README.md
├── models/                    # Model weights and class mappings
│   ├── README.md              # Teammate export & placement guide
│   ├── class_names.json       # JSON array of ordered target classes
│   └── cropguard_model.keras  # [Placed here by training pipeline]
└── tests/                     # Automated test suite
    └── test_model_utils.py    # Unit tests for preprocessing, labels, and inference
```

---

## 🚀 Getting Started & Setup

### 1. Prerequisites
Ensure you have **Python 3.10, 3.11, or 3.12** installed.

Verify your installation:
```powershell
python --version
pip --version
```

### 2. Environment Setup (Recommended)

#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🖥️ Launching the Application

Start the Streamlit development server:

```powershell
streamlit run app.py
```

The web dashboard will automatically open in your default browser at:
👉 **`http://localhost:8501`**

---

## 🧠 Model Integration Guide (For Teammates)

When your teammate finishes training the model in Google Colab or Kaggle:

### Step 1: Export Model from Colab / Keras
```python
# In your Colab notebook:
model.save("cropguard_model.keras")
```

### Step 2: Export Class Names
```python
import json

# Extract class names in the exact order of the softmax output layer:
class_names = train_ds.class_names  # or list(train_generator.class_indices.keys())

with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)
```

### Step 3: Place Files in CropGuard
Copy the exported files into the `models/` directory:
* `models/cropguard_model.keras`
* `models/class_names.json`

*(Note: CropGuard also automatically detects `cropguard_model.keras` in the project root directory).*

### Step 4: Verify Preprocessing Mode
Check the normalization used during training and select it in `config.py` or directly from the **Inference Pipeline** sidebar in the web app:
* `scaled_0_1`: If divided by 255.0.
* `mobilenet_v2`: If using `tf.keras.applications.mobilenet_v2.preprocess_input`.
* `raw_0_255`: If model contains an internal `Rescaling(1./255)` layer.

---

## 🧪 Running Automated Tests

Run the full pytest suite to verify preprocessing shapes, label parsing, and mock model inference:

```powershell
pytest tests/ -v
```

---

## 🛠️ Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **"Trained Model Awaiting Integration"** | No model file found in `models/` or root | Place `cropguard_model.keras` in `models/`. The app will auto-detect and activate real inference. |
| **"Class count mismatch" Error** | Model output units do not match `class_names.json` length | Ensure `class_names.json` has the exact same number of classes as the model's output layer. |
| **`ModuleNotFoundError: No module named 'tensorflow'`** | TensorFlow is not installed in the active environment | Run `pip install tensorflow` (requires Python 3.10–3.12). |
| **Blank / Gray image preview** | Corrupt or unsupported image file | Upload valid JPG, JPEG, or PNG format leaf images. |

---

## ⚖️ Academic Disclaimer
CropGuard is an **educational research prototype** created for college deep learning coursework. All classifications and confidence values are algorithmic outputs and should not be considered certified phytosanitary diagnoses. Always consult an agricultural extension specialist or agronomist for critical farming decisions.
