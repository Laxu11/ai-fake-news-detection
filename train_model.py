import os
import re
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = Path("dataset/fake_news.csv")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_data(df):
    df = df.copy()
    df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["combined_text"] = df["combined_text"].apply(clean_text)
    df["label"] = df["label"].astype(int)
    return df


def train_model():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df = prepare_data(df)

    X = df["combined_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
