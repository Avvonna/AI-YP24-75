from app.configs.base_config import BaseModelConfig
from app.core.data_manager import DataManager
from app.trainers import AutoARIMATrainer, BaseModelTrainer, CatBoostTrainer, LSTMTrainer


class ModelRegistry:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.models = {
            "auto_arima": AutoARIMATrainer,
            "catboost": CatBoostTrainer,
            "lstm": LSTMTrainer,
        }

    def get_available_models(self) -> list[str]:
        return list(self.models.keys())

    def get_trainer(self, model_name: str) -> BaseModelTrainer:
        if model_name not in self.models:
            raise ValueError(f"Модель '{model_name}' не поддерживается.")
        return self.models[model_name](self.data_manager)

    def get_config(self, model_name: str) -> type[BaseModelConfig]:
        if model_name not in self.models:
            raise ValueError(f"Модель '{model_name}' не поддерживается.")
        return self.models[model_name].config_class

    def get_config_schema(self, model_name: str):
        config_cls = self.get_config(model_name)
        return config_cls.model_json_schema()
