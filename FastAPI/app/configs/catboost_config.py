from pydantic import Field

from .base_config import BaseModelConfig


class CatBoostConfig(BaseModelConfig):
    model_type: str = "catboost"

    iterations: int = Field(
        default=100,
        ge=1,
        description="Количество итераций (деревьев) в модели CatBoost"
    )

    learning_rate: float = Field(
        default=0.1,
        ge=0.0001,
        le=1.0,
        description="Скорость обучения (learning rate) — чем меньше, тем стабильнее, но дольше"
    )

    depth: int = Field(
        default=6,
        ge=1,
        le=16,
        description="Глубина деревьев (обычно от 4 до 10); влияет на переобучение"
    )
