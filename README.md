# AI Fake News Detection System

An end-to-end machine learning project that detects fake news using a classic NLP pipeline based on TF-IDF and Logistic Regression. The application includes a training pipeline and a Streamlit user interface for real-time predictions.

## Features
- High-accuracy fake news classification using TF-IDF + Logistic Regression
- Clean text preprocessing pipeline
- Easy model retraining with CSV datasets
- Web app for single-text prediction
- Production-friendly project structure

## Project Structure
```text
ai-fake-news-detection/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── dataset/
│   └── fake_news.csv
├── model/
│   └── fake_news_model.pkl
└── .gitignore
```

## Tech Stack
- Python 3.10+
- scikit-learn
- pandas
- Streamlit
- joblib

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset
The project expects a CSV file named `dataset/fake_news.csv` with at least these columns:
- `title`
- `text`
- `label`

Label convention:
- `0` = real news
- `1` = fake news

## Training the Model
```bash
python train_model.py
```

This script:
- loads the dataset
- cleans and combines article text
- trains a TF-IDF + Logistic Regression model
- evaluates accuracy
- saves the model to `model/fake_news_model.pkl`

## Running the Web App
```bash
streamlit run app.py
```

## Usage
1. Open the local Streamlit app in your browser.
2. Paste a headline or article text.
3. Click "Analyze".
4. View the prediction and confidence scores.

## Model Notes
This is a strong baseline for fake news detection and works well for educational and demo projects. For production-scale systems, consider:
- larger and cleaner datasets
- BERT/transformer-based models
- source credibility analysis
- fact-checking integration
- API deployment for real-time use

## License
MIT
