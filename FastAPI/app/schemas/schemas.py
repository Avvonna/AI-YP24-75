from datetime import datetime
from typing import Optional

import numpy as np
from app.configs import ModelConfigUnion
from pydantic import BaseModel, Field, field_validator


class TimeSeriesData(BaseModel):
    """Модель временного ряда"""
    dates: list[datetime] = Field(..., description="Список дат временного ряда")
    values: list[float] = Field(..., description="Список значений временного ряда")
    experiment_name: str = Field(..., description="Уникальное имя эксперимента")

    @field_validator("values")
    def validate_values(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("Список значений не может быть пустым")
        if not all(isinstance(x, (int, float)) or np.isnan(x) for x in v):
            raise ValueError("Все значения должны быть числовыми")
        return v

class ModelSelectRequest(BaseModel):
    """Запрос на выбор модели"""
    model_name: str = Field(..., description="Название модели")

class PredictRequest(BaseModel):
    """Запрос на прогнозирование"""
    ticker: str = Field(..., description="Название тикера")
    base_date: str = Field(..., description="Дата для прогноза")
    forecast_period: int = Field(default=10, ge=1, description="Горизонт прогнозирования")
    config: Optional[ModelConfigUnion] = None

    @field_validator("forecast_period")
    def validate_forecast_period(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Период прогнозирования должен быть положительным")
        return v

class HistoricalDataRequest(BaseModel):
    """Запрос исторических данных"""
    start_date: Optional[str] = Field(..., description="Начальная дата")
    end_date: Optional[str] = Field(..., description="Конечная дата")

    @field_validator("start_date", "end_date", mode="before")
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if not v or v.strip() == "":
            return None
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError as e:
            raise ValueError("Неверный формат. Используйте YYYY-MM-DD") from e

    @field_validator("end_date", mode="after")
    def validate_date_range(cls, v: Optional[str], info) -> Optional[str]:
        start_date = info.data.get("start_date")
        if not v or not start_date:
            return v
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(v)
        if end <= start:
            raise ValueError("Конечная дата должна быть больше начальной.")
        return v

class ExperimentComparisonRequest(BaseModel):
    """Запрос на сравнение экспериментов"""
    experiment_names: list[str] = Field(..., min_length=1, description="Список имён экспериментов для сравнения")

    @field_validator("experiment_names")
    def validate_experiment_names(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("Дублирование имён экспериментов не допускается")
        return v
