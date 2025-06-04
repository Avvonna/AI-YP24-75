import numpy as np
import pandas as pd
from app.services.feature_engineering import preprocess_for_model
from app.services.trainers.base_trainer import BaseModelTrainer
from catboost import CatBoostRegressor


class CatBoostTrainer(BaseModelTrainer):
    def __init__(self):
        self.model = None
        self.last_known_features = None
        self.target_column = "value"

    def train(self, data, config):
        df = pd.DataFrame({
            "date": pd.to_datetime(data.dates),
            self.target_column: data.values
        }).set_index("date")

        df, new_cols = preprocess_for_model(
            df,
            target_column=self.target_column,
            return_new_colnames=True
        )

        X = df[new_cols]
        y = df[self.target_column]

        model_params = getattr(config, "model_params", {})
        self.model = CatBoostRegressor(**model_params, verbose=0)
        self.model.fit(X, y)

        self.last_known_features = X.iloc[-1:].copy()

        mae = float(np.mean(np.abs(self.model.predict(X) - y)))
        return self.model, {"mae": mae}

    def predict(self, steps: int):
        if self.model is None or self.last_known_features is None:
            raise ValueError("Model not trained or features not stored")

        preds = []
        current_features = self.last_known_features.copy()

        for _ in range(steps):
            pred = self.model.predict(current_features)[0]
            preds.append(pred)

            # Простая симуляция обновления фичей
            current_features = current_features.copy()
            if f"{self.target_column}_lag_1" in current_features.columns:
                current_features[f"{self.target_column}_lag_1"] = pred

        return preds
