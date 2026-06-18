import html
import json

import streamlit as st

from src.config import (
    CONFIDENCE_DECIMALS,
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    MODEL_METRICS_PATH,
)


def apply_custom_styles():
    st.markdown(
        """
        <style>
        :root {
            --ink: #17212b;
            --muted: #5f6b76;
            --border: #dce2e7;
            --surface: #ffffff;
            --page: #f5f7f8;
            --teal: #0f766e;
            --teal-soft: #eaf6f3;
            --blue-soft: #eef4ff;
            --amber-soft: #fff7e6;
        }

        .stApp {
            background: var(--page);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, p, span, div {
            letter-spacing: 0;
        }

        h1 {
            color: var(--ink);
            margin-bottom: 0.25rem;
        }

        .showcase-subtitle {
            color: var(--muted);
            font-size: 1.08rem;
            line-height: 1.6;
            margin: 0 0 1rem;
        }

        .instruction-box {
            background: var(--teal-soft);
            border: 1px solid #b9ddd7;
            border-left: 4px solid var(--teal);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 0.75rem 0 1.25rem;
        }

        .instruction-title {
            color: var(--ink);
            font-weight: 700;
            margin-bottom: 0.65rem;
        }

        .instruction-steps {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
        }

        .instruction-step {
            color: #334155;
            line-height: 1.45;
        }

        .step-number {
            align-items: center;
            background: var(--teal);
            border-radius: 999px;
            color: white;
            display: inline-flex;
            font-size: 0.82rem;
            font-weight: 700;
            height: 1.55rem;
            justify-content: center;
            margin-right: 0.4rem;
            width: 1.55rem;
        }

        .supported-section {
            margin-bottom: 1.5rem;
        }

        .supported-heading {
            color: var(--ink);
            font-size: 0.98rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }

        .supported-copy {
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 0.65rem;
        }

        .chip-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .class-chip {
            background: var(--surface);
            border: 1px solid #cbd5e1;
            border-radius: 999px;
            color: #243142;
            display: inline-flex;
            font-size: 0.88rem;
            font-weight: 600;
            padding: 0.38rem 0.72rem;
            transition: background 120ms ease, border-color 120ms ease;
        }

        .class-chip:hover {
            background: var(--blue-soft);
            border-color: #8fb0e8;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: 0 8px 22px rgba(23, 33, 43, 0.06);
        }

        div[data-testid="stButton"] button {
            border-radius: 6px;
            font-weight: 650;
            min-height: 2.75rem;
        }

        button[kind="primary"] {
            background: var(--teal) !important;
            border-color: var(--teal) !important;
        }

        .drawing-tip {
            color: var(--muted);
            font-size: 0.86rem;
            margin: 0.75rem 0 0;
        }

        .prediction-empty {
            background: #f8fafc;
            border: 1px dashed #bdc7d2;
            border-radius: 8px;
            color: var(--muted);
            margin: 1rem 0;
            padding: 1rem;
            text-align: center;
        }

        .mode-badge {
            border-radius: 999px;
            display: inline-flex;
            font-size: 0.82rem;
            font-weight: 700;
            margin: 0.1rem 0 0.75rem;
            padding: 0.32rem 0.62rem;
        }

        .mode-trained {
            background: #e7f6ed;
            border: 1px solid #a9d8ba;
            color: #166534;
        }

        .mode-fallback {
            background: var(--amber-soft);
            border: 1px solid #e8c879;
            color: #854d0e;
        }

        div[data-testid="stMetric"] {
            padding: 0.2rem 0;
        }

        @media (max-width: 720px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1.25rem;
            }

            .instruction-steps {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_header():
    st.title("SketchGuess AI")
    st.markdown(
        '<p class="showcase-subtitle">'
        "Draw a simple doodle and let the AI guess the top 3 possible objects."
        "</p>",
        unsafe_allow_html=True,
    )


def display_instructions():
    st.markdown(
        """
        <div class="instruction-box">
            <div class="instruction-title">How to play</div>
            <div class="instruction-steps">
                <div class="instruction-step">
                    <span class="step-number">1</span>
                    Choose one of the supported doodle categories
                </div>
                <div class="instruction-step">
                    <span class="step-number">2</span>
                    Draw it on the pad
                </div>
                <div class="instruction-step">
                    <span class="step-number">3</span>
                    Click Predict and see the AI's top guesses
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_supported_classes(class_names):
    chips = "".join(
        f'<span class="class-chip">{html.escape(class_name.title())}</span>'
        for class_name in class_names
    )

    st.markdown(
        f"""
        <div class="supported-section">
            <div class="supported-heading">Supported doodles</div>
            <div class="supported-copy">Choose one of these categories for the best results.</div>
            <div class="chip-list">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_model_metrics():
    if not MODEL_METRICS_PATH.exists():
        return None

    try:
        return json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def display_model_info(is_using_trained_model):
    if is_using_trained_model:
        badge_class = "mode-trained"
        status_text = "Using trained model"
    else:
        badge_class = "mode-fallback"
        status_text = "Using dummy fallback"

    st.markdown(
        f'<span class="mode-badge {badge_class}">{status_text}</span>',
        unsafe_allow_html=True,
    )

    metrics = load_model_metrics()

    with st.expander("Model info"):
        if is_using_trained_model:
            st.write("The app is using the saved TensorFlow/Keras model.")
        else:
            st.write("No trained model is loaded, so the app is using dummy predictions.")

        if metrics is None:
            st.write("No saved training metrics found yet.")
            return

        st.write(f"**Classes:** {metrics['number_of_classes']}")
        st.write(f"**Samples per class:** {metrics['samples_per_class']}")
        st.write(f"**Training epochs:** {metrics['epochs']}")
        st.write(f"**Test accuracy:** {metrics['test_accuracy']:.2%}")
        st.write(f"**Test loss:** {metrics['test_loss']:.4f}")


def display_top_predictions(predictions):
    st.subheader("Top guesses")

    for label, confidence in predictions:
        safe_confidence = min(max(confidence, CONFIDENCE_MIN), CONFIDENCE_MAX)
        confidence_text = f"{safe_confidence:.{CONFIDENCE_DECIMALS}%}"

        st.write(f"**{label.title()}** - {confidence_text}")
        st.progress(safe_confidence)


def display_empty_predictions():
    st.markdown(
        '<div class="prediction-empty">Your top 3 guesses will appear here after you draw and predict.</div>',
        unsafe_allow_html=True,
    )


def display_session_score(correct_count, total_count):
    accuracy = correct_count / total_count if total_count else 0

    score_col_1, score_col_2, score_col_3 = st.columns(3)

    with score_col_1:
        st.metric("Attempts", total_count)

    with score_col_2:
        st.metric("Correct", correct_count)

    with score_col_3:
        st.metric("Accuracy", f"{accuracy:.0%}")
