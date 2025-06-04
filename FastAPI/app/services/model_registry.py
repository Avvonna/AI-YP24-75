from app.services.trainers.arima_trainer import AutoARIMATrainer
from app.services.trainers.catboost_trainer import CatBoostTrainer

# from app.services.trainers.lstm_trainer import LSTMTrainer

class ModelRegistry:
    def __init__(self):
        self.models = {
            "auto_arima": AutoARIMATrainer,
            "catboost": CatBoostTrainer,
            # "lstm": LSTMTrainer
        }

    def get_trainer(self, model_name: str):
        if model_name not in self.models:
            raise ValueError(f"Модель '{model_name}' не поддерживается.")
        return self.models[model_name]()
