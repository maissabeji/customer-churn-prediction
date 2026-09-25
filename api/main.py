from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import load_pipeline, predict_single

# Create the FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)

# Load model ONCE at startup — not on every request
# Loading a model file on every request would make the API 10-100x slower
pipeline = load_pipeline()


# Input schema — defines exactly what JSON the /predict endpoint accepts
# Every field name must match what predict_single() and the pipeline expect
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float


# Output schema — defines exactly what JSON the /predict endpoint returns
class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str


# Endpoint 1: health check — lets monitoring systems verify the API is alive
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}


# Endpoint 2: prediction — the actual business logic
@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    try:
        result = predict_single(customer.model_dump(), pipe=pipeline)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))