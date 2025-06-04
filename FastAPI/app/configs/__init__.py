from typing import Union

from .arima_config import AutoARIMAConfig
from .catboost_config import CatBoostConfig
from .lstm_config import LSTMConfig

ModelConfigUnion = Union[AutoARIMAConfig, CatBoostConfig, LSTMConfig]

__all__ = [
    "AutoARIMAConfig",
    "CatBoostConfig",
    "LSTMConfig",
    "ModelConfigUnion"
]
