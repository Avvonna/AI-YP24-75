from typing import Any
import pandas as pd
from app.models.schemas import HistoricalDataRequest
from app.services.ml_service import ml_service
from fastapi import APIRouter, HTTPException, UploadFile, File

router = APIRouter(prefix="/api/tickers", tags=["Tickers"])


@router.get(
        "/",
        response_model=list[str],
        description="Возвращает список доступных тикеров"
        )
async def get_tickers():
    try:
        return ml_service.get_available_tickers()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
        "/{ticker}/history",
        response_model=dict[str, Any],
        description="Возвращает исторические данные тикера")
async def get_ticker_history(ticker: str, request: HistoricalDataRequest):
    try:
        return ml_service.get_ticker_history(
            ticker,
            request.start_date,
            request.end_date
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
        "/",
        response_model=dict[str, str],
        description="Добавляет или обновляет данные тикера")
async def add_ticker_data(ticker: str, file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    try:
        ml_service.add_ticker_data(ticker, df)
        return {"status": "ticker added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete(
        "/{ticker}",
        response_model=dict[str, str],
        description="Удаляет данные тикера"
        )
async def remove_ticker(ticker: str):
    try:
        ml_service.remove_ticker(ticker)
        return {"status": "ticker removed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
