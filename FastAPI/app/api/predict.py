# FastAPI/app/api/predict.py

from typing import Annotated, Any

from app.models.schemas import CurrentModelPredictRequest, PredictRequest
from app.services.ml_service import ml_service

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/predict", tags=["Prediction"])

@router.post("/", response_model=dict[str, Any])
async def predict(request: Annotated[PredictRequest, "Прогноз по тикеру"]):
    try:
        return ml_service.train_and_predict(
            request.ticker,
            request.base_date,
            request.forecast_period,
            config=request.config if hasattr(request, "config") else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.post("/current", response_model=dict[str, Any])
async def predict_current_model(request: Annotated[CurrentModelPredictRequest, "Прогноз текущей модели"]):
    try:
        forecast, intervals = ml_service.predict_current_model(
            request.data,
            request.steps,
            request.config
        )
        return {
            "forecast": forecast,
            "confidence_intervals": intervals
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
