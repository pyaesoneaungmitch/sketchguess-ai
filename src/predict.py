import json

import numpy as np

from src.config import CLASS_NAMES_PATH, MODEL_PATH, TOP_K_PREDICTIONS


DUMMY_PREDICTIONS = [
    ("cat", 0.62),
    ("dog", 0.25),
    ("car", 0.13),
]

_MODEL_ARTIFACTS = None


def load_model():
    global _MODEL_ARTIFACTS

    if _MODEL_ARTIFACTS is not None:
        return _MODEL_ARTIFACTS

    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        return None

    try:
        # Delay TensorFlow import so the dummy fallback stays lightweight.
        import tensorflow as tf

        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        _MODEL_ARTIFACTS = {
            "model": model,
            "class_names": class_names,
        }
        return _MODEL_ARTIFACTS
    except Exception as error:
        print(f"Could not load trained model, using dummy predictions instead: {error}")
        return None


def predict_top_k(model_artifacts, image_array, top_k=TOP_K_PREDICTIONS):
    if model_artifacts is None:
        return DUMMY_PREDICTIONS[:top_k]

    model = model_artifacts["model"]
    class_names = model_artifacts["class_names"]

    probabilities = model.predict(image_array, verbose=0)[0]
    # NumPy sorts ascending, so take the final scores and reverse their order.
    top_indices = np.argsort(probabilities)[-top_k:][::-1]

    return [
        (class_names[index], float(probabilities[index]))
        for index in top_indices
    ]
