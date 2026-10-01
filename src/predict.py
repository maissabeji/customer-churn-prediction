"""
Model packaging and inference for churn prediction.
Saves and loads the fitted pipeline, exposes predict_single() for the API.
"""
import joblib
import pandas as pd
from pathlib import Path
from src.feature_engineering import BINARY_COLS

MODEL_PATH = Path("models/churn_pipeline.pkl")
THRESHOLD = 0.35  # chosen from threshold tuning in Milestone 7

def save_pipeline(pipe, path: Path = MODEL_PATH) -> None:
    # your code here — 2 lines
    # hint: Path.mkdir(parents=True, exist_ok=True) then joblib.dump()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    print(f"Model saved to {path}")


def load_pipeline(path: Path = MODEL_PATH):
    # your code here — 1 line
    return joblib.load(path)
    

def predict_single(customer_data: dict, pipe=None) -> dict:
    """
    Predict churn for one customer.
    Returns churn_probability, churn_prediction (0/1), risk_level (LOW/MEDIUM/HIGH).
    """
    # your code here:
    # 1. load pipeline if not passed in
    # 2. convert customer_data dict to DataFrame (one row)
    # 3. get probability with predict_proba
    # 4. apply THRESHOLD to get binary prediction
    # 5. return dict with churn_probability, churn_prediction, risk_level
    # risk_level: HIGH if proba >= 0.5, MEDIUM if >= 0.35, LOW otherwise
    if pipe is None:
        pipe = load_pipeline()
    df = pd.DataFrame([customer_data])
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = (df[col] == 'Yes').astype(int)
    proba = pipe.predict_proba(df)[0][1]
    prediction = int(proba >= THRESHOLD)

    if proba >= 0.5:
        risk_level = "HIGH"
    elif proba >= 0.35:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "churn_probability": round(float(proba), 4),
        "churn_prediction": prediction,
        "risk_level": risk_level
    }    