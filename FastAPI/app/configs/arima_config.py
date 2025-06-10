from typing import Optional

from app.configs.base_config import BaseModelConfig
from pydantic import Field


class AutoARIMAConfig(BaseModelConfig):
    model_type: str = "auto_arima"

    max_p: int = Field(
        default=5,
        ge=0,
        description="Максимальное значение порядка AR (авторегрессии)"
    )

    max_d: int = Field(
        default=5,
        ge=0,
        description="Максимальный порядок интеграции (разностей) для стационарности"
    )

    max_q: int = Field(
        default=5,
        ge=0,
        description="Максимальный порядок MA (скользящего среднего)"
    )

    max_P: int = Field(
        default=6,
        ge=0,
        description="Максимальный сезонный порядок AR (авторегрессии)"
    )

    max_D: int = Field(
        default=6,
        ge=0,
        description="Максимальный сезонный порядок разностей"
    )

    max_Q: int = Field(
        default=6,
        ge=0,
        description="Максимальный сезонный порядок MA (скользящего среднего)"
    )

    seasonal: bool = Field(
        default=True,
        description="Учитывать ли сезонность в модели"
    )

    seasonal_period: Optional[int] = Field(
        default=7,
        ge=1,
        description="Период сезонности (например, 7 для недельных данных)"
    )

    stepwise: bool = Field(
        default=True,
        description="Использовать ли пошаговый (stepwise) режим подбора параметров"
    )
