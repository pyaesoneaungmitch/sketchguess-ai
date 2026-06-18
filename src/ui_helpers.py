import json

import streamlit as st

from src.config import (
    CONFIDENCE_DECIMALS,
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    MODEL_METRICS_PATH,
)


def display_instructions():
    st.write("Draw a simple doodle, then ask the app to guess what it is.")


def load_model_metrics():
    if not MODEL_METRICS_PATH.exists():
        return None

    try:
        return json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def display_model_info(is_using_trained_model):
    if is_using_trained_model:
        st.caption("Prediction mode: trained model")
    else:
        st.caption("Prediction mode: dummy fallback")

    metrics = load_model_metrics()

    with st.expander("Model info"):
        if is_using_trained_model:
            st.write("The app is using the saved TensorFlow/Keras model.")
        else:
            st.write("No trained model is loaded, so the app is using dummy predictions.")

        if metrics is None:
            st.write("No saved training metrics found yet.")
            return

        st.write(f"Classes: {metrics['number_of_classes']}")
        st.write(f"Samples per class: {metrics['samples_per_class']}")
        st.write(f"Training epochs: {metrics['epochs']}")
        st.write(f"Test accuracy: {metrics['test_accuracy']:.2%}")
        st.write(f"Test loss: {metrics['test_loss']:.4f}")


def display_top_predictions(predictions):
    st.subheader("Top guesses")

    for label, confidence in predictions:
        safe_confidence = min(max(confidence, CONFIDENCE_MIN), CONFIDENCE_MAX)
        confidence_text = f"{safe_confidence:.{CONFIDENCE_DECIMALS}%}"

        st.write(f"**{label.title()}** - {confidence_text}")
        st.progress(safe_confidence)


def display_session_score(correct_count, total_count):
    accuracy = correct_count / total_count if total_count else 0

    score_col_1, score_col_2, score_col_3 = st.columns(3)

    with score_col_1:
        st.metric("Correct", correct_count)

    with score_col_2:
        st.metric("Total", total_count)

    with score_col_3:
        st.metric("Accuracy", f"{accuracy:.0%}")
