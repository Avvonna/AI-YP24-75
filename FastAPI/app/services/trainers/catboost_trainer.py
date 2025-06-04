import numpy as np
import pandas as pd
from app.models.base_model import BaseModelTrainer
from catboost import CatBoostRegressor


class CatBoostTrainer(BaseModelTrainer):
    def __init__(self):
        self.model = None

    def train(self, data, config):
        df = pd.DataFrame({
            "date": pd.to_datetime(data.dates),
            "value": data.values
        })
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year

        X = df[["day", "month", "year"]]
        y = df["value"]

        self.model = CatBoostRegressor(verbose=0)
        self.model.fit(X, y)

        return self.model, {
            "mae": float(np.mean(np.abs(self.model.predict(X) - y)))
        }

    def predict(self, steps: int):
        # Простейший пример предсказания (будущее — просто последние даты + шаги)
        raise NotImplementedError("Нужно реализовать forecast logic")
