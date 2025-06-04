from app.models.base_model import BaseModelConfig
from pydantic import Field


class CatBoostConfig(BaseModelConfig):
    iterations: int = Field(default=100)
    learning_rate: float = Field(default=0.1)
    depth: int = Field(default=6)
