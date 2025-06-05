import logging
from typing import Annotated, Any

from app.models.schemas import CurrentModelPredictRequest, PredictRequest
from app.services.ml_service import ml_service

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])

@router.post("/", response_model=dict[str, Any])
async def predict(request: Annotated[PredictRequest, "Прогноз по тикеру"]):
    try:
        logger.info(f"[predict/] Получен запрос на прогноз: {request}")
        result = ml_service.train_and_predict(
            request.ticker,
            request.base_date,
            request.forecast_period,
            config=request.config if hasattr(request, "config") else None
        )
        logger.info(f"[predict/] Прогноз успешно выполнен для тикера {request.ticker}")
        return result
    except Exception as e:
        logger.exception(f"[predict/] Ошибка при прогнозировании тикера {request.ticker}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.post("/current", response_model=dict[str, Any])
async def predict_current_model(request: Annotated[CurrentModelPredictRequest, "Прогноз текущей модели"]):
    try:
        logger.info(f"[predict/current] Получен запрос на прогноз текущей модели. "
                    f"Шагов: {request.steps}, Длина данных: {len(request.data)}")
        forecast, intervals = ml_service.predict_current_model(
            request.data,
            request.steps,
            request.config
        )
        logger.info("[predict/current] Прогноз текущей модели успешно выполнен")
        return {
            "forecast": forecast,
            "confidence_intervals": intervals
        }
    except Exception as e:
        logger.exception("[predict/current] Ошибка при прогнозе текущей модели")
        raise HTTPException(status_code=500, detail=str(e)) from e
