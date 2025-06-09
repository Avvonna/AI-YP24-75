from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelSelectSchema(BaseModel):
    model_name: str

class ModelSelectResponseSchema(BaseModel):
    status: Literal["Модель выбрана"]
    model_name: str = Field(..., examples=["catboost"])

class ModelSchemaResponse(BaseModel):
    model_name: str
    config_schema: dict[str, Any]

class AvailableModelsResponse(BaseModel):
    models: list[str] = Field(..., example=["auto_arima", "catboost", "lstm"])
