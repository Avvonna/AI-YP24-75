from typing import Optional

from app.models.base_model import BaseModelConfig
from pydantic import Field


class AutoARIMAConfig(BaseModelConfig):
    seasonal: bool = Field(default=True)
    seasonal_period: Optional[int] = Field(default=7, ge=1)
    max_p: int = 5
    max_d: int = 5
    max_q: int = 5
    max_P: int = 6
    max_D: int = 6
    max_Q: int = 6
