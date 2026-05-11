import streamlit as st
import pickle
import re
import pandas as pd
import os


def clean_text(text):
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return text


st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

if not os.path.exists("sentiment_model.pkl") or not os.path.exists("vectorizer.pkl"):
    st.error("Model files not found. First run: python train_model.py")
    st.stop()

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.title("🎬 IMDB Movie Review Sentiment Analysis")

    review = st.text_area("Enter Movie Review:", height=180)

    if st.button("Analyze Sentiment"):
        if review.strip() == "":
            st.warning("Please enter a movie review.")
        else:
            cleaned_review = clean_text(review)
            review_vector = vectorizer.transform([cleaned_review])

            prediction = model.predict(review_vector)[0]
            probabilities = model.predict_proba(review_vector)[0]

            st.session_state.review = review
            st.session_state.prediction = prediction
            st.session_state.probabilities = probabilities

            st.session_state.page = "result"
            st.rerun()

elif st.session_state.page == "result":
    st.title("📊 Sentiment Result")

    prediction = st.session_state.prediction
    probabilities = st.session_state.probabilities

    if prediction == "positive":
        st.success("Positive Review 😊")
    else:
        st.error("Negative Review 😞")

    st.subheader("Your Review")
    st.write(st.session_state.review)

    st.subheader("Sentiment Probability Graph")

    graph_data = pd.DataFrame({
        "Sentiment": model.classes_,
        "Probability": probabilities
    })

    st.bar_chart(graph_data, x="Sentiment", y="Probability")

    st.subheader("Probability Values")

    for sentiment, prob in zip(model.classes_, probabilities):
        st.write(f"{sentiment.capitalize()}: {prob * 100:.2f}%")

    if st.button("Analyze Another Review"):
        st.session_state.page = "home"
        st.rerun()