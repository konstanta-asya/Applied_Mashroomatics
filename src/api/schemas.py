from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class EdibilityStatus(str, Enum):
    EDIBLE = "edible"
    INEDIBLE = "inedible"
    POISONOUS = "poisonous"
    DEADLY = "deadly"
    UNKNOWN = "unknown"


class MetadataInput(BaseModel):
    """Optional metadata to improve predictions"""
    month: Optional[int] = Field(None, ge=1, le=12, description="Month (1-12)")
    habitat: Optional[str] = Field(None, description="Habitat type")
    substrate: Optional[str] = Field(None, description="Substrate type")


class SpeciesPrediction(BaseModel):
    """Single species prediction"""
    species_id: int
    species_name: str
    confidence: float = Field(..., ge=0, le=1)
    common_name_ua: Optional[str] = None
    edibility: EdibilityStatus = EdibilityStatus.UNKNOWN


class PredictionResponse(BaseModel):
    """Full prediction response"""
    success: bool
    predictions: list[SpeciesPrediction]
    top_prediction: SpeciesPrediction
    is_confident: bool = Field(..., description="True if top confidence > threshold")
    warning: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "predictions": [
                    {
                        "species_id": 42,
                        "species_name": "Amanita muscaria",
                        "confidence": 0.85,
                        "common_name_ua": "Мухомор червоний",
                        "edibility": "poisonous"
                    }
                ],
                "top_prediction": {
                    "species_id": 42,
                    "species_name": "Amanita muscaria",
                    "confidence": 0.85,
                    "common_name_ua": "Мухомор червоний",
                    "edibility": "poisonous"
                },
                "is_confident": True,
                "warning": "This mushroom is POISONOUS!"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    num_classes: int


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    detail: Optional[str] = None