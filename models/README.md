# Models Directory

This directory is where your trained deep learning model and class label mapping should be placed.

## Expected Files

1. **`cropguard_model.keras`** (or `cropguard_model.h5`):
   - Your trained TensorFlow / Keras deep learning model.
   - Example architecture: MobileNetV2, ResNet50, EfficientNet, or custom CNN trained on leaf datasets.
   - Exported from Google Colab using:
     ```python
     model.save("cropguard_model.keras")
     ```

2. **`class_names.json`**:
   - A JSON list containing the ordered class names corresponding to your model's softmax output layer.
   - Example format:
     ```json
     [
       "Apple___Apple_scab",
       "Apple___Black_rot",
       "Apple___healthy",
       "Tomato___Early_blight",
       "Tomato___healthy"
     ]
     ```
   - In Google Colab / Keras, this list can be exported directly from your `image_dataset_from_directory` or `ImageDataGenerator`:
     ```python
     import json
     # For tf.keras.utils.image_dataset_from_directory:
     class_names = train_ds.class_names
     with open("class_names.json", "w") as f:
         json.dump(class_names, f, indent=2)
     ```

## Preprocessing Modes

In `config.py`, configure the `DEFAULT_PREPROCESSING_MODE` that matches your Colab training pipeline:

- `"scaled_0_1"` (Default): Scales raw pixels `[0, 255]` to `[0.0, 1.0]` via division by 255.0.
- `"mobilenet_v2"`: Scales pixels to `[-1.0, 1.0]` using `tf.keras.applications.mobilenet_v2.preprocess_input`.
- `"raw_0_255"`: Passes raw pixels directly `[0, 255]` if your model includes an internal `Rescaling` layer.

## Validation Rule

The application will verify that:
$$\text{len(model output vector)} == \text{len(class\_names.json)}$$
If there is a mismatch, the UI will present an actionable error message highlighting the exact count difference.
