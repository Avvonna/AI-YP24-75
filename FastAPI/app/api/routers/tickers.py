import io
import logging

import pandas as pd
from app.core.ml_service import ml_pipeline
from app.schemas import HistoricalDataRequest, TickerHistory
from pydantic import BaseModel

from fastapi import APIRouter, HTTPException, UploadFile

router = APIRouter(prefix="/api/tickers", tags=["Tickers"])
logger = logging.getLogger(__name__)


class StatusResponse(BaseModel):
    status: str


@router.get(
    "/",
    response_model=list[str],
    description="Возвращает список доступных тикеров"
)
async def get_tickers():
    try:
        return ml_pipeline.get_available_tickers()
    except Exception as e:
        logger.exception("Ошибка при получении списка тикеров")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/{ticker}/history",
    response_model=TickerHistory,
    description="Возвращает исторические данные тикера"
)
async def get_ticker_history(ticker: str, request: HistoricalDataRequest):
    try:
        return ml_pipeline.get_ticker_history(
            ticker,
            request.start_date,
            request.end_date
        )
    except Exception as e:
        logger.exception(f"Ошибка при получении истории по тикеру {ticker}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/",
    response_model=StatusResponse,
    description="Добавляет или обновляет данные тикера"
)
async def add_ticker_data(ticker: str, file: UploadFile):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        if "date" not in df.columns:
            raise ValueError("Отсутствует обязательная колонка 'date'")

        if ticker.upper() not in map(str.upper, df.columns):
            raise ValueError(f"Не найдена колонка с данными по тикеру '{ticker}'")

        ml_pipeline.add_ticker_data(ticker, df)
        return StatusResponse(status="ticker added")

    except ValueError as ve:
        logger.warning(f"Ошибка валидации CSV: {ve}")
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        logger.exception(f"Ошибка при добавлении данных для тикера {ticker}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete(
    "/{ticker}",
    response_model=StatusResponse,
    description="Удаляет данные тикера"
)
async def remove_ticker(ticker: str):
    try:
        ml_pipeline.remove_ticker(ticker)
        return StatusResponse(status="ticker removed")
    except Exception as e:
        logger.exception(f"Ошибка при удалении тикера {ticker}")
        raise HTTPException(status_code=500, detail=str(e)) from e
