from pydantic import Field

from app.configs.base_config import BaseModelConfig


class CatBoostConfig(BaseModelConfig):
    iterations: int = Field(default=100)
    learning_rate: float = Field(default=0.1)
    depth: int = Field(default=6)
