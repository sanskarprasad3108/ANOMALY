from fastapi import FastAPI
from fastapi.responses import FileResponse
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()

# Load ML objects
model = load_model("autoencoder_model.keras")
scaler = joblib.load("scaler.pkl")
threshold = joblib.load("threshold.pkl")
baseline_sample = joblib.load("baseline.pkl")
feature_names = joblib.load("features.pkl")

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/detect")
def detect(feature_name: str, feature_value: float):

    if feature_name not in feature_names:
        return {"error": "Feature not found"}

    sample = baseline_sample.copy()
    sample[feature_name] = feature_value

    df = pd.DataFrame([sample], columns=feature_names)
    scaled = scaler.transform(df)

    recon = model.predict(scaled)
    error = float(np.mean((scaled - recon) ** 2))

    result = "Anomaly" if error > threshold else "Normal"

    return {
        "feature": feature_name,
        "value": feature_value,
        "error": error,
        "result": result
    }
