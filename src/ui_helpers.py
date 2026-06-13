import streamlit as st

from src.config import CONFIDENCE_DECIMALS, CONFIDENCE_MAX, CONFIDENCE_MIN


def display_instructions():
    st.write("Draw a simple doodle, then ask the app to guess what it is.")


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
