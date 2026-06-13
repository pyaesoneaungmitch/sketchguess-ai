import streamlit as st
from streamlit_drawable_canvas import st_canvas

from src.config import CANVAS_SIZE
from src.predict import load_model, predict_top_k
from src.preprocessing import preprocess_canvas_image
from src.ui_helpers import (
    display_instructions,
    display_session_score,
    display_top_predictions,
)


st.set_page_config(page_title="SketchGuess AI", layout="centered")


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


initialize_session_state()

st.title("SketchGuess AI")
display_instructions()

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

clear_col, predict_col = st.columns(2)

with clear_col:
    st.button("Clear", use_container_width=True, on_click=clear_canvas)

with predict_col:
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    processed_image = preprocess_canvas_image(canvas_result.image_data)

    if processed_image is None:
        st.session_state.predictions = []
        st.session_state.pending_feedback = False
        st.warning("Draw something on the canvas before predicting.")
    else:
        model = load_model()
        st.session_state.predictions = predict_top_k(model, processed_image)
        st.session_state.pending_feedback = True

if st.session_state.predictions:
    display_top_predictions(st.session_state.predictions)

    st.subheader("Was the prediction correct?")
    feedback_col_1, feedback_col_2 = st.columns(2)

    with feedback_col_1:
        st.button(
            "Correct",
            use_container_width=True,
            disabled=not st.session_state.pending_feedback,
            on_click=record_feedback,
            args=(True,),
        )

    with feedback_col_2:
        st.button(
            "Wrong",
            use_container_width=True,
            disabled=not st.session_state.pending_feedback,
            on_click=record_feedback,
            args=(False,),
        )

display_session_score(
    st.session_state.correct_count,
    st.session_state.total_count,
)
