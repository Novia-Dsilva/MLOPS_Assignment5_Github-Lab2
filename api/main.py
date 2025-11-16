"""
FastAPI application for Housing Price Prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import glob
import os
import sys
from typing import List, Optional
import json

sys.path.insert(0, os.path.abspath('..'))

from src.preprocessing import HousingPreprocessor

# Initialize FastAPI app
app = FastAPI(
    title="Housing Price Prediction API",
    description="API for predicting California housing prices using ML models",
    version="1.0.0"
)

# Global variables for model and preprocessor
model = None
preprocessor = None
model_info = {}


class HousingFeatures(BaseModel):
    """Input features for prediction"""
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms per household")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average number of household members")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")
    
    class Config:
        schema_extra = {
            "example": {
                "MedInc": 3.5,
                "HouseAge": 25.0,
                "AveRooms": 5.5,
                "AveBedrms": 1.2,
                "Population": 1200.0,
                "AveOccup": 3.0,
                "Latitude": 34.05,
                "Longitude": -118.25
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response"""
    predicted_price: float = Field(..., description="Predicted house price in $100k")
    predicted_price_usd: float = Field(..., description="Predicted house price in USD")
    model_version: str
    model_type: str


class ModelInfo(BaseModel):
    """Model information"""
    model_version: str
    model_type: str
    timestamp: str
    n_features: int
    available_models: List[str]


def load_latest_model():
    """Load the most recent model"""
    global model, preprocessor, model_info
    
    # Find all model files
    model_files = glob.glob('models/model_*_*.joblib')
    
    if not model_files:
        raise FileNotFoundError("No trained models found!")
    
    # Sort by timestamp (assuming format: model_TIMESTAMP_TYPE.joblib)
    model_files.sort(reverse=True)
    latest_model_path = model_files[0]
    
    # Extract timestamp and model type
    filename = os.path.basename(latest_model_path)
    parts = filename.replace('model_', '').replace('.joblib', '').split('_')
    timestamp = parts[0]
    model_type = '_'.join(parts[1:])
    
    # Load model
    model = joblib.load(latest_model_path)
    
    # Load preprocessor
    preprocessor_path = f'models/preprocessor_{timestamp}.pkl'
    preprocessor = HousingPreprocessor.load_preprocessor(preprocessor_path)
    
    # Load feature info
    feature_info_path = f'models/feature_info_{timestamp}.json'
    if os.path.exists(feature_info_path):
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)
    else:
        feature_info = {'n_features': 'unknown'}
    
    model_info = {
        'model_version': latest_model_path,
        'model_type': model_type,
        'timestamp': timestamp,
        'n_features': feature_info.get('n_features', 'unknown'),
        'available_models': [os.path.basename(f) for f in model_files]
    }
    
    print(f" Loaded model: {latest_model_path}")
    return model, preprocessor


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_latest_model()
        print("🚀 FastAPI server started successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load model on startup: {e}")
        print("Model will be loaded on first prediction request")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Housing Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "model_info": "/model_info",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return {"status": "unhealthy", "message": "Model not loaded"}
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_info": model_info
    }


@app.get("/model_info", response_model=ModelInfo)
async def get_model_info():
    """Get current model information"""
    if model is None:
        load_latest_model()
    
    return ModelInfo(**model_info)


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HousingFeatures):
    """
    Predict house price for given features
    
    Returns predicted price in $100k units and USD
    """
    global model, preprocessor
    
    # Load model if not loaded
    if model is None or preprocessor is None:
        load_latest_model()
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Preprocess
        input_features = preprocessor.create_features(input_data)
        input_processed = preprocessor.transform(input_features)
        
        # Predict
        prediction = model.predict(input_processed)[0]
        prediction_usd = prediction * 100000  # Convert to USD
        
        return PredictionResponse(
            predicted_price=float(prediction),
            predicted_price_usd=float(prediction_usd),
            model_version=model_info['model_version'],
            model_type=model_info['model_type']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(features_list: List[HousingFeatures]):
    """
    Predict house prices for multiple inputs
    
    Returns list of predictions
    """
    global model, preprocessor
    
    # Load model if not loaded
    if model is None or preprocessor is None:
        load_latest_model()
    
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([f.dict() for f in features_list])
        
        # Preprocess
        input_features = preprocessor.create_features(input_data)
        input_processed = preprocessor.transform(input_features)
        
        # Predict
        predictions = model.predict(input_processed)
        
        results = [
            {
                "predicted_price": float(pred),
                "predicted_price_usd": float(pred * 100000),
                "model_version": model_info['model_version'],
                "model_type": model_info['model_type']
            }
            for pred in predictions
        ]
        
        return {"predictions": results, "count": len(results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/reload_model")
async def reload_model():
    """Reload the latest model"""
    try:
        load_latest_model()
        return {
            "message": "Model reloaded successfully",
            "model_info": model_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)