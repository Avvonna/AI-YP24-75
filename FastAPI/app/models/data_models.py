from datetime import datetime
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator


class TimeSeriesData(BaseModel):
    """Модель временного ряда"""
    dates: list[datetime] = Field(
        ...,
        description="Список дат временного ряда"
    )
    values: list[float] = Field(
        ...,
        description="Список значений временного ряда"
    )
    experiment_name: str = Field(
        ...,
        description="Уникальное имя эксперимента"
    )

    @field_validator("values")
    def validate_values(cls, v: list[float]) -> list[float]:
        """Проверка значений временного ряда"""
        if not v:
            raise ValueError("Список значений не может быть пустым")
        if not all(isinstance(x, (int, float)) or np.isnan(x) for x in v):
            raise ValueError("Все значения должны быть числовыми")
        return v

class ModelConfig(BaseModel):
    """Параметры конфигурации модели"""
    seasonal: bool = Field(
        default=True,
        description="Учитывать ли сезонность"
    )
    seasonal_period: Optional[int] = Field(
        default=7,
        ge=1,
        description="Период сезонности"
    )
    max_p: int = Field(
        default=5,
        description="Максимальный порядок AR члена"
    )

    @field_validator("seasonal_period")
    def validate_seasonal_period(cls, v: Optional[int], values: dict[str, Any]) -> Optional[int]:
        """Проверка периода сезонности"""
        if values.get("seasonal") and (v is None or v < 1):
            raise ValueError("Период сезонности должен быть положительным числом при seasonal=True")
        return v

class TickerDataRequest(BaseModel):
    """Запрос данных тикера"""
    ticker: str = Field(..., description="Название тикера")
    start_date: str = Field(..., description="Начальная дата")
    end_date: str = Field(..., description="Конечная дата")

class ModelSelectRequest(BaseModel):
    """Запрос на выбор модели"""
    model_name: str = Field(..., description="Название модели")

class PredictRequest(BaseModel):
    """Запрос на прогнозирование"""
    ticker: str = Field(..., description="Название тикера")
    base_date: str = Field(..., description="Дата для прогноза")
    forecast_period: int = Field(default=10, description="Горизонт прогнозирования")

class PlotRequest(BaseModel):
    """Запрос на построение графика"""
    ticker: str = Field(..., description="Название тикера")
    start_date: str = Field(..., description="Начальная дата")
    end_date: str = Field(..., description="Конечная дата")

class ExperimentComparisonRequest(BaseModel):
    """Запрос на сравнение экспериментов"""
    experiment_names: set[str] = Field(
        ...,
        description="Набор имён экспериментов для сравнения"
    )
