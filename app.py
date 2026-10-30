import pickle
import re
from pathlib import Path

import streamlit as st

MODEL_PATH = Path("model/fake_news_model.pkl")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


st.set_page_config(page_title="AI Fake News Detection", page_icon="📰", layout="centered")
st.title("AI Fake News Detection System")
st.write("Paste a headline or article text to estimate whether it is likely real or fake.")

article = st.text_area("News Text", height=220, placeholder="Enter article title or full content here...")

if st.button("Analyze", use_container_width=True):
    if not article.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        try:
            model = load_model()
            cleaned = clean_text(article)
            prediction = model.predict([cleaned])[0]
            probabilities = model.predict_proba([cleaned])[0]

            if prediction == 1:
                result = "Likely Fake News"
                color = "#d9534f"
            else:
                result = "Likely Real News"
                color = "#2ca02c"

            st.markdown(f"<h3 style='color:{color}; margin-bottom: 10px;'>{result}</h3>", unsafe_allow_html=True)

            fake_probability = round(probabilities[1] * 100, 2)
            real_probability = round(probabilities[0] * 100, 2)

            st.write(f"Probability of Fake News: {fake_probability}%")
            st.write(f"Probability of Real News: {real_probability}%")

        except FileNotFoundError:
            st.error("The trained model was not found. Please run `python train_model.py` first.")
