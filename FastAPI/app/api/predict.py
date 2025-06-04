import logging
from typing import Annotated, Any

from app.models.schemas import PredictRequest
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
