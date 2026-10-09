"""
predict.py
----------
Inference module for Garbage Classification CNN.
Supports model loading, single image preprocessing, batch inference,
and structured prediction output with confidence scores and eco-tips.
"""

import os
from pathlib import Path
from typing import Dict, Union, List
import numpy as np
from PIL import Image
import tensorflow as tf

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

RECYCLING_GUIDANCE = {
    'cardboard': {
        'category': 'Recyclable Paperboard',
        'action': 'Recycle in Paper & Cardboard Bin',
        'tip': 'Flatten boxes to save space. Remove heavy tape or wax coatings.'
    },
    'glass': {
        'category': 'Recyclable Glass',
        'action': 'Recycle in Glass Container Bin',
        'tip': 'Rinse thoroughly to remove food or liquid residues. Remove metal caps.'
    },
    'metal': {
        'category': 'Recyclable Metal / Aluminum',
        'action': 'Recycle in Metal Bin',
        'tip': 'Empty cans completely. Crush aluminum cans to optimize bin volume.'
    },
    'paper': {
        'category': 'Recyclable Clean Paper',
        'action': 'Recycle in Paper Bin',
        'tip': 'Ensure paper is dry and clean. Do not recycle food-stained paper.'
    },
    'plastic': {
        'category': 'Recyclable Plastic',
        'action': 'Recycle in Plastics Bin',
        'tip': 'Rinse containers. Check resin codes (#1 PET and #2 HDPE are widely accepted).'
    },
    'trash': {
        'category': 'Non-Recyclable Waste',
        'action': 'Dispose in General Waste Bin',
        'tip': 'Non-recyclable or composite waste. Place securely in landfill waste bag.'
    }
}

_LOADED_MODEL = None


def load_model(model_path: str = "models/best_garbage_cnn.keras") -> tf.keras.Model:
    """
    Loads saved Keras model artifact with caching.
    """
    global _LOADED_MODEL
    abs_path = Path(model_path).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"Model file not found at '{abs_path}'. Please run training first.")

    if _LOADED_MODEL is None:
        _LOADED_MODEL = tf.keras.models.load_model(str(abs_path))
    return _LOADED_MODEL


def preprocess_image(image_input: Union[str, Path, bytes, Image.Image], target_size=(224, 224)) -> np.ndarray:
    """
    Preprocesses input image into standard (1, 224, 224, 3) tensor with MobileNetV2 preprocessing.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, bytes):
        import io
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input format.")

    img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_preprocessed, axis=0)


def predict_image(
    image_input: Union[str, Path, bytes, Image.Image],
    model_path: str = "models/best_garbage_cnn.keras"
) -> Dict:
    """
    Executes garbage classification prediction on a single input image.

    Returns:
        dict: Containing predicted_class, confidence, probabilities, and guidance.
    """
    model = load_model(model_path)
    tensor = preprocess_image(image_input)
    
    preds = model.predict(tensor, verbose=0)[0]
    best_idx = int(np.argmax(preds))
    confidence = float(preds[best_idx])
    predicted_class = CLASSES[best_idx]

    probabilities = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}
    sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)

    guidance = RECYCLING_GUIDANCE.get(predicted_class, {
        'category': 'Unknown',
        'action': 'Inspect manually',
        'tip': 'Check local waste disposal guidelines.'
    })

    return {
        'predicted_class': predicted_class,
        'confidence': round(confidence * 100, 2),
        'confidence_raw': confidence,
        'probabilities': probabilities,
        'top_predictions': [
            {'class': cls_name, 'prob': round(prob * 100, 2)}
            for cls_name, prob in sorted_probs
        ],
        'guidance': guidance
    }


if __name__ == "__main__":
    import sys
    print("[INFO] Predict module ready.")
