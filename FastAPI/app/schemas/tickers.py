from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class TickerHistory(BaseModel):
    ticker: str
    dates: list[date]
    values: list[float]

class HistoricalDataRequest(BaseModel):
    """Запрос исторических данных"""
    start_date: Optional[date] = Field(..., description="Начальная дата", examples=["2022-12-30"])
    end_date: Optional[date] = Field(..., description="Конечная дата", examples=["2024-12-30"])

    @model_validator(mode="after")
    def check_date_order(cls, values):
        start = values.start_date
        end = values.end_date
        if (start and end) and (end <= start):
            raise ValueError("Конечная дата должна быть позже начальной")
        return values
