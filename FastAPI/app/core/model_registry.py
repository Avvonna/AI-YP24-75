from app.core import DataManager
from app.trainers import AutoARIMATrainer, CatBoostTrainer, LSTMTrainer


class ModelRegistry:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.models = {
            "auto_arima": AutoARIMATrainer(data_manager),
            "catboost": CatBoostTrainer(data_manager),
            "lstm": LSTMTrainer(data_manager),
        }

    def get_trainer(self, model_name: str):
        if model_name not in self.models:
            raise ValueError(f"Модель '{model_name}' не поддерживается.")
        return self.models[model_name]
