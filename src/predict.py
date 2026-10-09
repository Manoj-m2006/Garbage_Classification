"""
predict.py
----------
Inference module for Garbage Classification CNN.
Supports lightweight TFLite inference for production serverless deployment (Vercel)
and full Keras model inference for local development.
"""

import os
import io
from pathlib import Path
from typing import Dict, Union, List
import numpy as np
from PIL import Image

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
_LOADED_INTERPRETER = None


def get_tflite_interpreter(tflite_path="models/best_garbage_cnn.tflite"):
    global _LOADED_INTERPRETER
    if _LOADED_INTERPRETER is not None:
        return _LOADED_INTERPRETER

    # Try tflite_runtime first (lightweight for Vercel)
    try:
        import tflite_runtime.interpreter as tflite
        _LOADED_INTERPRETER = tflite.Interpreter(model_path=tflite_path)
    except ImportError:
        try:
            import tensorflow.lite as tflite
            _LOADED_INTERPRETER = tflite.Interpreter(model_path=tflite_path)
        except Exception as e:
            _LOADED_INTERPRETER = None

    if _LOADED_INTERPRETER is not None:
        _LOADED_INTERPRETER.allocate_tensors()

    return _LOADED_INTERPRETER


def preprocess_image_numpy(image_input: Union[str, Path, bytes, Image.Image], target_size=(224, 224)) -> np.ndarray:
    """
    Preprocesses input image into standard (1, 224, 224, 3) MobileNetV2 tensor using pure NumPy.
    Formula: (img / 127.5) - 1.0
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input format.")

    img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    img_preprocessed = (img_array / 127.5) - 1.0
    return np.expand_dims(img_preprocessed, axis=0)


def predict_image(
    image_input: Union[str, Path, bytes, Image.Image],
    model_path: str = "models/best_garbage_cnn.tflite"
) -> Dict:
    """
    Executes garbage classification prediction on a single input image.
    Tries lightweight TFLite inference first, with fallback to full TensorFlow Keras if available.
    """
    tensor = preprocess_image_numpy(image_input)
    preds = None

    # Option 1: TFLite Inference (Ultra-lightweight for Vercel Serverless)
    tflite_file = "models/best_garbage_cnn.tflite"
    if os.path.exists(tflite_file):
        interpreter = get_tflite_interpreter(tflite_file)
        if interpreter is not None:
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], tensor)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]

    # Option 2: Fallback to Keras model if TFLite not available
    if preds is None:
        keras_file = "models/best_garbage_cnn.keras"
        if os.path.exists(keras_file):
            try:
                import tensorflow as tf
                global _LOADED_MODEL
                if _LOADED_MODEL is None:
                    _LOADED_MODEL = tf.keras.models.load_model(keras_file)
                preds = _LOADED_MODEL.predict(tensor, verbose=0)[0]
            except Exception:
                pass

    if preds is None:
        raise RuntimeError("No trained model file found or model execution failed.")

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
    print("[INFO] Predict module ready.")
