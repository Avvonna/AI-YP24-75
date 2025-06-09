from datetime import date
from typing import Optional

from app.configs import ModelConfigUnion
from pydantic import BaseModel, Field

from .tickers import TickerHistory


class ForecastRequestSchema(BaseModel):
    ticker: str = Field(..., examples=["LKOH"])
    base_date: date = Field(..., examples=["2024-01-30"])
    forecast_period: int = Field(default=10, ge=1)
    config: Optional[ModelConfigUnion] = None

class ForecastConfidenceIntervals(BaseModel):
    lower: list[float]
    upper: list[float]

class ForecastResult(BaseModel):
    experiment_id: str
    forecast_dates: list[date]
    forecast_values: list[float]
    confidence_intervals: Optional[ForecastConfidenceIntervals]
