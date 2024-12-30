import sys
from pathlib import Path
from typing import Annotated, Any, Optional

import uvicorn

from fastapi import FastAPI, HTTPException

sys.path.append(str(Path(__file__).parent.parent))

from app.models.data_models import (
    ExperimentComparisonRequest,
    ModelConfig,
    ModelSelectRequest,
    PlotRequest,
    PredictRequest,
    TickerDataRequest,
    TimeSeriesData,
)
from app.services.ml_service import MLService

app = FastAPI(
    title="Stock price prediction",
    description="by team 75",
    version="1.0.0"
)

ml_service = MLService()

@app.post(
    "/api/ticker-data",
    response_model=dict[str, Any],
    responses={
        500: {"description": "Внутренняя ошибка сервера"}
    },
    summary="Получение данных тикера"
)
async def get_ticker_data(
    request: Annotated[TickerDataRequest, "Параметры запроса данных тикера"]
) -> dict[str, Any]:
    """Получение исторических данных тикера."""
    try:
        return ml_service.get_ticker_data(request.ticker, request.start_date, request.end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post(
    "/api/model/select",
    response_model=dict[str, str],
    responses={
        400: {"description": "Неверные параметры запроса"}
    },
    summary="Выбор модели"
)
async def select_model(
    request: Annotated[ModelSelectRequest, "Параметры выбора модели"]
) -> dict[str, str]:
    """Выбор модели для прогнозирования."""
    try:
        return ml_service.set_model(request.model_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

@app.post(
    "/api/predict",
    response_model=dict[str, Any],
    responses={
        500: {"description": "Внутренняя ошибка сервера"}
    },
    summary="Прогнозирование"
)
async def predict(
    request: Annotated[PredictRequest, "Параметры прогнозирования"]
) -> dict[str, Any]:
    """Прогнозирование значений тикера."""
    try:
        return ml_service.train_and_predict(
            request.ticker,
            request.base_date,
            request.forecast_period
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post(
    "/api/plot",
    response_model=dict[str, Any],
    responses={
        500: {"description": "Внутренняя ошибка сервера"}
    },
    summary="График тикера"
)
async def plot_data(
    request: Annotated[PlotRequest, "Параметры построения графика"]
) -> dict[str, Any]:
    """Получение данных для построения графика тикера."""
    try:
        return ml_service.get_ticker_data(
            request.ticker,
            request.start_date,
            request.end_date
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post(
    "/api/plot-prediction",
    response_model=dict[str, Any],
    responses={
        500: {"description": "Внутренняя ошибка сервера"}
    },
    summary="График прогноза"
)
async def plot_prediction(
    request: Annotated[PredictRequest, "Параметры прогноза"]
) -> dict[str, Any]:
    """Получение данных для построения графика прогноза."""
    try:
        return ml_service.train_and_predict(
            request.ticker,
            request.base_date,
            request.forecast_period
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post(
    "/api/train",
    response_model=dict[str, Any],
    responses={
        500: {"description": "Внутренняя ошибка сервера"},
        422: {"description": "Ошибка валидации"}
    },
    summary="Обучение новой модели"
)
async def train_model(
    data: Annotated[TimeSeriesData, "Данные временного ряда для обучения"],
    config: Annotated[ModelConfig, "Параметры настройки модели"]
) -> dict[str, Any]:
    """Обучение модели и возврат метрик качества."""
    try:
        metrics = ml_service.train_model(data, config)
        return {
            "status": "успешно",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post(
    "/api/compare",
    response_model=dict[str, dict],
    responses={
        200: {"description": "Успешное сравнение экспериментов"}
    },
    summary="Сравнение экспериментов"
)
async def compare_experiments(
    request: Annotated[
        ExperimentComparisonRequest,
        "Параметры сравнения экспериментов"
    ]
) -> dict[str, dict]:
    """Сравнение метрик разных экспериментов."""
    try:
        return ml_service.compare_experiments(request.experiment_names)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post(
    "/api/model/save",
    response_model=dict[str, str],
    responses={
        200: {"description": "Модель успешно сохранена"}
    },
    summary="Сохранение модели"
)
async def save_model(
    path: Annotated[str, "Путь для сохранения модели"],
    experiment_name: Optional[str] = None,
    format_type: str = "pickle"
) -> dict[str, str]:
    """Сохранение обученной модели в файл."""
    try:
        ml_service.save_model(path, experiment_name, format_type)
        return {"status": "Модель успешно сохранена"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post(
    "/api/model/load",
    response_model=dict[str, str],
    responses={
        200: {"description": "Модель успешно загружена"}
    },
    summary="Загрузка модели"
)
async def load_model(
    path: Annotated[str, "Путь к сохраненной модели"],
    experiment_name: Annotated[str, "Имя для загружаемого эксперимента"],
    format_type: str = "pickle"
) -> dict[str, str]:
    """Загрузка сохраненной модели из файла."""
    try:
        ml_service.load_model(path, experiment_name, format_type)
        return {"status": "Модель успешно загружена"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
