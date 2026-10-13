from pathlib import Path
import json

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Locate project files
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_catboost.cbm"
METADATA_PATH = BASE_DIR / "models" / "churn_metadata.json"

# Load metadata
with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

THRESHOLD = metadata["decision_threshold"]
FEATURE_COLUMNS = metadata["feature_columns"]

# Load the trained model
model = CatBoostClassifier()
model.load_model(str(MODEL_PATH))

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained CatBoost model.",
    version="1.0.0",
)


class CustomerData(BaseModel):
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
    TotalCharges_clean: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": metadata["model_name"],
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        customer_data = customer.model_dump()

        # Preserve the model's expected feature order
        input_df = pd.DataFrame(
            [[customer_data[col] for col in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )

        churn_probability = float(model.predict_proba(input_df)[0][1])
        prediction = int(churn_probability >= THRESHOLD)

        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(churn_probability, 4),
            "decision_threshold": THRESHOLD,
            "model_name": metadata["model_name"],
            "note": (
                "This is a model prediction, not a guarantee "
                "that the customer will or will not churn."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc