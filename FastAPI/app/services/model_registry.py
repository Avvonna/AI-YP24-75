from app.services.data_manager import DataManager
from app.services.trainers.arima_trainer import AutoARIMATrainer
from app.services.trainers.catboost_trainer import CatBoostTrainer
from app.services.trainers.lstm_trainer import LSTMTrainer

# from app.services.trainers.lstm_trainer import LSTMTrainer

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
