from typing import Any, Optional

from app.configs import ModelConfigUnion
from pydantic import BaseModel, Field, field_validator

from .tickers import TickerHistory


class ExperimentMetrics(BaseModel):
    aic: Optional[float] = None
    bic: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None

class ExperimentRecord(BaseModel):
    name: str
    model: Any
    config: ModelConfigUnion
    metrics: ExperimentMetrics
    training_data: TickerHistory

class ComparisonResult(BaseModel):
    experiments: list[ExperimentRecord]
    missing_experiments: list[str]

class ExperimentComparisonSchema(BaseModel):
    experiment_names: list[str] = Field(..., min_length=1)

    @field_validator("experiment_names")
    def no_duplicates(cls, v):
        if len(v) != len(set(v)):
            raise ValueError("Список содержит дубликаты")
        return v
