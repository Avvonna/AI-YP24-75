import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from app.models.data_models import (
   TimeSeriesData,
   ModelConfig, 
   ExperimentComparisonRequest,
   ModelSelectRequest,
   PredictRequest,
   CurrentModelPredictRequest,
   HistoricalDataRequest
)
from app.services.ml_service import MLService

def setup_logging():
   log_dir = Path("logs")
   log_dir.mkdir(exist_ok=True)
   
   formatter = logging.Formatter(
       '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   
   file_handler = RotatingFileHandler(
       log_dir / "app.log",
       maxBytes=10*1024*1024, 
       backupCount=5,
       encoding='utf-8'
   )
   file_handler.setFormatter(formatter)
   
   console_handler = logging.StreamHandler()
   console_handler.setFormatter(formatter)
   
   logger = logging.getLogger()
   logger.setLevel(logging.INFO)
   logger.addHandler(file_handler)
   logger.addHandler(console_handler)
   
   return logger

logger = setup_logging()

app = FastAPI(
   title="123",
   description="223",
   version="1.0.0" 
)

ml_service = MLService()

@app.get("/api/tickers", response_model=List[str])
async def get_tickers() -> List[str]:
   """Get list of available tickers"""
   logger.info("Запрос списка тикеров")
   return ml_service.get_available_tickers()

@app.get("/api/tickers/{ticker}/history")
async def get_ticker_histor(
   ticker: str,
   request: HistoricalDataRequest
) -> Dict[str, Any]:
   """Get historical data for ticker"""
   try:
       logger.info(f"Получение истории {ticker}")
       return ml_service.get_historical_data(
           ticker,
           request.start_date,
           request.end_date
       )
   except Exception as e:
       logger.error(f"Ошибка получения истории: {str(e)}")
       raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tickers/add", response_model=Dict[str, str])
async def add_ticker(ticker: str) -> Dict[str, str]:
   """Add new ticker"""
   try:
       logger.info(f"Добавление тикера {ticker}")
       ml_service.add_ticker(ticker)
       return {"status": "success"}
   except Exception as e:
       logger.error(f"Ошибка добавления тикера: {str(e)}")
       raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/tickers/{ticker}", response_model=Dict[str, str])
async def remove_ticker(ticker: str) -> Dict[str, str]:
   """Remove ticker"""
   try:
       logger.info(f"Удаление тикера {ticker}")
       ml_service.remove_ticker(ticker)
       return {"status": "success"}
   except Exception as e:
       logger.error(f"Ошибка удаления тикера: {str(e)}")
       raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/select", response_model=Dict[str, str])
async def select_model(
   request: Annotated[ModelSelectRequest, "Параметры выбора модели"]
) -> Dict[str, str]:
   """Select forecasting model"""
   try:
       logger.info(f"Выбор модели {request.model_name}")
       return ml_service.set_model(request.model_name)
   except Exception as e:
       logger.error(f"Ошибка выбора модели: {str(e)}")
       raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict", response_model=Dict[str, Any])
async def predict(
   request: Annotated[PredictRequest, "Параметры прогнозирования"]
) -> Dict[str, Any]:
   """Get prediction for ticker"""
   try:
       logger.info(f"Прогнозирование для тикера {request.ticker}")
       return ml_service.train_and_predict(
           request.ticker,
           request.base_date,
           request.forecast_period
       )
   except Exception as e:
       logger.error(f"Ошибка прогнозирования: {str(e)}")
       raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/predict", response_model=Dict[str, Any])
async def predict_current_model(
   request: Annotated[CurrentModelPredictRequest, "Параметры прогноза текущей модели"]
) -> Dict[str, Any]:
   """Get prediction using current model"""
   try:
       logger.info(f"Прогноз current_model на {request.steps} шагов")
       forecast, intervals = ml_service.predict_current_model(
           request.data, 
           request.steps
       )
       return {
           "forecast": forecast,
           "confidence_intervals": intervals
       }
   except Exception as e:
       logger.error(f"Ошибка прогноза current_model: {str(e)}")
       raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/experiments/compare")
async def compare_experiments(
   request: ExperimentComparisonRequest
) -> Dict[str, Any]:
   """Compare multiple experiments"""
   try:
       logger.info("Сравнение экспериментов")
       return ml_service.compare_experiments(request.experiment_names)
   except Exception as e:
       logger.error(f"Ошибка сравнения экспериментов: {str(e)}")
       raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000)