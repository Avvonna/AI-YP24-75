from typing import Literal, Optional

from app.configs.base_config import BaseModelConfig
from pydantic import Field


class AutoARIMAConfig(BaseModelConfig):
    model_type: Literal["auto_arima"] = "auto_arima"
    max_p: int = 5
    max_d: int = 5
    max_q: int = 5
    max_P: int = 6
    max_D: int = 6
    max_Q: int = 6
    seasonal: bool = Field(default=True)
    seasonal_period: Optional[int] = Field(default=7, ge=1)
