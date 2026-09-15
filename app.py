import os
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Union, Literal
from sklearn.base import BaseEstimator, TransformerMixin

# Define FeatureEngineer class matching training definition
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        X['Total_Services_Count'] = 0
        for s in services:
            if s in X.columns:
                X['Total_Services_Count'] += (X[s] == 'Yes').astype(int)
        
        expected_total = X['tenure'] * X['MonthlyCharges']
        X['Monthly_To_Total_Ratio'] = np.where(
            X['tenure'] > 0, 
            X['TotalCharges'] / (expected_total + 1e-5), 
            1.0
        )
        
        X['Tenure_Group'] = pd.cut(
            X['tenure'], 
            bins=[-1, 12, 24, 48, 72], 
            labels=['0-12m', '12-24m', '24-48m', '48-72m']
        ).astype(str)
        
        return X

# Bind FeatureEngineer to __main__ module to ensure joblib unpickling compatibility
setattr(sys.modules['__main__'], 'FeatureEngineer', FeatureEngineer)

# Initialize FastAPI App
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="REST API for predicting customer churn probability using a Decision Tree Classifier.",
    version="1.0.0"
)

# Define Model Path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "churn_model.pkl")

# Global model variable
model_pipeline = None

@app.on_event("startup")
def load_model():
    global model_pipeline
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Saved model file not found at {MODEL_PATH}. Please train and save the model first.")
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Model successfully loaded from {MODEL_PATH}")

# Pydantic Schema for Input Validation
class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, description="Tenure in months (must be >= 0)")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly charges in USD")
    TotalCharges: Union[float, str] = Field(..., description="Total charges in USD (float or numeric string)")

    @field_validator("TotalCharges")
    @classmethod
    def validate_total_charges(cls, v):
        if isinstance(v, str):
            v_str = v.strip()
            if v_str == "":
                return 0.0
            try:
                val = float(v_str)
                if val < 0:
                    raise ValueError("TotalCharges must be >= 0")
                return val
            except ValueError:
                raise ValueError("TotalCharges must be a valid numeric value or empty string")
        elif isinstance(v, (int, float)):
            if v < 0:
                raise ValueError("TotalCharges must be >= 0")
            return float(v)
        raise ValueError("Invalid format for TotalCharges")

class PredictionResponse(BaseModel):
    prediction: Literal["Yes", "No"]
    churn_probability: float

@app.get("/")
def root():
    return {
        "message": "Telco Customer Churn Prediction API is running.",
        "docs_url": "/docs",
        "predict_endpoint": "/predict"
    }

@app.get("/health")
def health():
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline is not initialized.")
    
    try:
        # Convert input dictionary to DataFrame
        data_dict = customer.dict()
        input_df = pd.DataFrame([data_dict])
        
        # Make prediction
        pred_int = model_pipeline.predict(input_df)[0]
        prob_float = float(model_pipeline.predict_proba(input_df)[0][1])
        
        prediction_str = "Yes" if pred_int == 1 else "No"
        
        return PredictionResponse(
            prediction=prediction_str,
            churn_probability=round(prob_float, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
