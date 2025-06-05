from typing import Literal

from app.configs.base_config import BaseModelConfig
from pydantic import Field


class CatBoostConfig(BaseModelConfig):
    model_type: Literal["catboost"] = "catboost"
    iterations: int = Field(default=100)
    learning_rate: float = Field(default=0.1)
    depth: int = Field(default=6)
