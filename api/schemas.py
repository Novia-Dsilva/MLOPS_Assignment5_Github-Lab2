"""
Pydantic schemas for API validation
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class HousingFeaturesBase(BaseModel):
    """Base schema for housing features"""
    MedInc: float = Field(..., ge=0, description="Median income (0-15)")
    HouseAge: float = Field(..., ge=0, le=100, description="House age in years")
    AveRooms: float = Field(..., ge=0, description="Average rooms per household")
    AveBedrms: float = Field(..., ge=0, description="Average bedrooms per household")
    Population: float = Field(..., ge=0, description="Population in block")
    AveOccup: float = Field(..., ge=0, description="Average occupancy")
    Latitude: float = Field(..., ge=32, le=42, description="Latitude")
    Longitude: float = Field(..., ge=-125, le=-114, description="Longitude")
    
    @validator('AveBedrms')
    def bedrooms_less_than_rooms(cls, v, values):
        if 'AveRooms' in values and v > values['AveRooms']:
            raise ValueError('Bedrooms cannot exceed total rooms')
        return v


class PredictionInput(HousingFeaturesBase):
    """Input for single prediction"""
    pass


class BatchPredictionInput(BaseModel):
    """Input for batch predictions"""
    features: List[HousingFeaturesBase]
    
    class Config:
        schema_extra = {
            "example": {
                "features": [
                    {
                        "MedInc": 3.5,
                        "HouseAge": 25.0,
                        "AveRooms": 5.5,
                        "AveBedrms": 1.2,
                        "Population": 1200.0,
                        "AveOccup": 3.0,
                        "Latitude": 34.05,
                        "Longitude": -118.25
                    }
                ]
            }
        }


class PredictionOutput(BaseModel):
    """Output for single prediction"""
    predicted_price_100k: float = Field(..., description="Predicted price in $100k")
    predicted_price_usd: float = Field(..., description="Predicted price in USD")
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    model_version: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class BatchPredictionOutput(BaseModel):
    """Output for batch predictions"""
    predictions: List[PredictionOutput]
    count: int
    average_price: float
    model_version: str


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    rmse: float
    mae: float
    r2_score: float
    mape: float
    timestamp: str


class HealthStatus(BaseModel):
    """API health status"""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: Optional[float]
    last_prediction: Optional[str]