
# fastapi_app.py - FastAPI for Housing Price Prediction
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import List, Union
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Housing Price Prediction API",
    description="Predict California housing prices using ML",
    version="1.0.0"
)

# Load model at startup
MODEL_PATH = 'housing_model_v1.joblib'
model = joblib.load(MODEL_PATH)
logger.info(f"Model loaded from {MODEL_PATH}")

# Pydantic model for input validation
class HouseFeatures(BaseModel):
    """Input schema for house features with validation."""
    MedInc: float = Field(..., gt=0, lt=20, description="Median income in block group")
    HouseAge: float = Field(..., gt=0, lt=100, description="Median house age")
    AveRooms: float = Field(..., gt=0, lt=100, description="Average rooms per household")
    AveBedrms: float = Field(..., gt=0, lt=50, description="Average bedrooms per household")
    Population: float = Field(..., gt=0, lt=50000, description="Block group population")
    AveOccup: float = Field(..., gt=0, lt=100, description="Average household occupancy")
    Latitude: float = Field(..., gt=32, lt=43, description="Latitude")
    Longitude: float = Field(..., lt=-114, gt=-125, description="Longitude")
    ocean_proximity: str = Field(..., description="Ocean proximity category")

    @validator('ocean_proximity')
    def validate_ocean(cls, v):
        allowed = ['NEAR_BAY', 'INLAND', '<1H_OCEAN', 'NEAR_OCEAN']
        if v not in allowed:
            raise ValueError(f'ocean_proximity must be one of {allowed}')
        return v

class PredictionResponse(BaseModel):
    """Output schema for predictions."""
    predictions: List[float]
    model_version: str
    timestamp: str

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "housing_v1",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: Union[HouseFeatures, List[HouseFeatures]]):
    """
    Predict house prices.
    Accepts single house or list of houses.
    """
    try:
        # Convert to DataFrame
        if isinstance(features, list):
            df = pd.DataFrame([f.dict() for f in features])
            logger.info(f"Batch prediction for {len(features)} houses")
        else:
            df = pd.DataFrame([features.dict()])
            logger.info(f"Single prediction for house")

        # Predict
        predictions = model.predict(df)

        return PredictionResponse(
            predictions=predictions.tolist(),
            model_version="v1",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/docs", include_in_schema=False)
async def custom_docs():
    """Redirect to Swagger UI."""
    return {"message": "Visit /docs for Swagger UI or /redoc for ReDoc"}

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
