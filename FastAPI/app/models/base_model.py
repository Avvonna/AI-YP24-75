from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseModelTrainer(ABC):
    @abstractmethod
    def train(self, data, config):
        pass

    @abstractmethod
    def predict(self, steps: int):
        pass

class BaseModelConfig(BaseModel):
    """Базовый класс для конфигураций моделей (ARIMA, CatBoost, LSTM и т.д.)"""
    model_type: str = "base"
