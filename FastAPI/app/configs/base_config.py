from pydantic import BaseModel


class BaseModelConfig(BaseModel):
    """Базовый класс для конфигураций моделей (ARIMA, CatBoost, LSTM и т.д.)"""
    model_type: str = "base"
