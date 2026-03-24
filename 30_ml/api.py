
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
from typing import List, Optional
import uvicorn

# Initialize app
app = FastAPI(
    title="House Price Prediction API",
    description="Production-grade house price prediction service",
    version="1.0.0"
)

# Load model at startup
model_artifact = joblib.load("capstone_model_v1.joblib")
pipeline = model_artifact['pipeline']
model = model_artifact['model']

# Define input schema based on Ames Housing features
class HouseFeatures(BaseModel):
    # Key numeric features (subset for demo)
    OverallQual: int = Field(..., ge=1, le=10, description="Overall material and finish quality")
    GrLivArea: float = Field(..., gt=0, description="Above grade living area square feet")
    TotalBsmtSF: float = Field(..., ge=0, description="Total basement square feet")
    GarageCars: float = Field(..., ge=0, description="Garage car capacity")
    YearBuilt: int = Field(..., ge=1872, le=2024, description="Original construction year")

    # Categorical features
    Neighborhood: str = Field(..., description="Physical locations within Ames city limits")
    HouseStyle: str = Field(..., description="Style of dwelling")

    # Optional additional features
    FullBath: Optional[float] = Field(2, ge=0, description="Full bathrooms above grade")
    BedroomAbvGr: Optional[float] = Field(3, ge=0, description="Bedrooms above grade")

class PredictionResponse(BaseModel):
    predicted_price: float
    price_log_scale: float
    model_version: str
    confidence: Optional[str] = None

@app.get("/")
def root():
    return {
        "message": "House Price Prediction API",
        "model_type": model_artifact['model_type'],
        "version": model_artifact['metadata']['version']
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metadata")
def get_metadata():
    return model_artifact['metadata']

@app.post("/predict", response_model=PredictionResponse)
def predict(house: HouseFeatures):
    try:
        # Convert to DataFrame (single row)
        import pandas as pd
        input_df = pd.DataFrame([house.dict()])

        # Preprocess
        X_processed = pipeline.transform(input_df)

        # Predict (returns log scale)
        pred_log = float(model.predict(X_processed)[0])
        pred_price = float(np.expm1(pred_log))

        return PredictionResponse(
            predicted_price=round(pred_price, 2),
            price_log_scale=round(pred_log, 4),
            model_version=model_artifact['metadata']['version'],
            confidence="based_on_holdout_r2_" + str(round(model_artifact['metrics']['holdout_r2'], 2))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(houses: List[HouseFeatures]):
    """Batch prediction endpoint for multiple houses."""
    try:
        import pandas as pd
        input_df = pd.DataFrame([h.dict() for h in houses])
        X_processed = pipeline.transform(input_df)
        preds_log = model.predict(X_processed)
        preds = np.expm1(preds_log)

        return {
            "predictions": [round(float(p), 2) for p in preds],
            "count": len(preds)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
