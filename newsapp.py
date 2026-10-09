
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="News Headline Classification API",
    description="API for predicting news categories from headlines",
    version="1.0"
)


# --------------------------------------------------
# Load saved model and preprocessing objects
# --------------------------------------------------

model = joblib.load("saved_model/final_model.pkl")
vectorizer = joblib.load("saved_model/tfidf_vectorizer.pkl")
label_encoder = joblib.load("saved_model/label_encoder.pkl")
model_type = joblib.load("saved_model/model_type.pkl")


# --------------------------------------------------
# Input format
# --------------------------------------------------

class HeadlineInput(BaseModel):
    headline: str


# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "News Headline Classification API",
        "model": model_type
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: HeadlineInput):

    # Clean headline
    headline = clean_text(data.headline)

    # Convert headline into TF-IDF features
    features = vectorizer.transform([headline])

    # Predict category
    prediction = model.predict(features)[0]

    # Convert numerical label back to category name
    category = label_encoder.inverse_transform([prediction])[0]

    return {
        "headline": data.headline,
        "predicted_category": category
    }
