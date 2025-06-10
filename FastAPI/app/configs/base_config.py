from pydantic import BaseModel


class BaseModelConfig(BaseModel):
    """Базовый класс для конфигураций моделей (ARIMA, CatBoost, LSTM и т.д.)"""
    model_type: str = "base"

    def get_model_params(self) -> dict:
        """Возвращает параметры модели в виде словаря"""
        return self.model_dump(exclude={"model_type"})
