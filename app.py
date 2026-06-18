import streamlit as st
from streamlit_drawable_canvas import st_canvas

from src.config import CANVAS_SIZE, CLASS_NAMES
from src.predict import load_model, predict_top_k
from src.preprocessing import preprocess_canvas_image
from src.ui_helpers import (
    apply_custom_styles,
    display_empty_predictions,
    display_header,
    display_instructions,
    display_model_info,
    display_session_score,
    display_supported_classes,
    display_top_predictions,
)


st.set_page_config(page_title="SketchGuess AI", layout="wide")


def initialize_session_state():
    defaults = {
        "canvas_key": 0,
        "predictions": [],
        "pending_feedback": False,
        "correct_count": 0,
        "total_count": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_canvas():
    st.session_state.canvas_key += 1
    st.session_state.predictions = []
    st.session_state.pending_feedback = False


def record_feedback(is_correct):
    st.session_state.total_count += 1

    if is_correct:
        st.session_state.correct_count += 1

    st.session_state.pending_feedback = False


apply_custom_styles()
initialize_session_state()

display_header()
display_instructions()
display_supported_classes(CLASS_NAMES)

model_artifacts = load_model()

left_column, right_column = st.columns([1.08, 0.92], gap="large")

with left_column:
    with st.container(border=True):
        st.subheader("Drawing pad")
        st.caption("Use the white pad below to sketch your object.")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=12,
            stroke_color="#111111",
            background_color="#ffffff",
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )

        predict_column, clear_column = st.columns([2, 1])

        with predict_column:
            predict_clicked = st.button(
                "Predict",
                type="primary",
                use_container_width=True,
            )

        with clear_column:
            st.button(
                "Clear",
                use_container_width=True,
                on_click=clear_canvas,
            )

        if predict_clicked:
            processed_image = preprocess_canvas_image(canvas_result.image_data)

            if processed_image is None:
                st.session_state.predictions = []
                st.session_state.pending_feedback = False
                st.warning("Draw something on the canvas before predicting.")
            else:
                st.session_state.predictions = predict_top_k(
                    model_artifacts,
                    processed_image,
                )
                st.session_state.pending_feedback = True

        st.markdown(
            '<p class="drawing-tip">Tip: draw one object clearly using simple lines.</p>',
            unsafe_allow_html=True,
        )

with right_column:
    with st.container(border=True):
        st.subheader("AI predictions")
        display_model_info(model_artifacts is not None)

        if st.session_state.predictions:
            display_top_predictions(st.session_state.predictions)

            st.subheader("Was the prediction correct?")
            correct_column, wrong_column = st.columns(2)

            with correct_column:
                st.button(
                    "Correct",
                    use_container_width=True,
                    disabled=not st.session_state.pending_feedback,
                    on_click=record_feedback,
                    args=(True,),
                )

            with wrong_column:
                st.button(
                    "Wrong",
                    use_container_width=True,
                    disabled=not st.session_state.pending_feedback,
                    on_click=record_feedback,
                    args=(False,),
                )
        else:
            display_empty_predictions()

        st.divider()
        st.subheader("Session score")
        display_session_score(
            st.session_state.correct_count,
            st.session_state.total_count,
        )
