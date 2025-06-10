import logging

import numpy as np
import pandas as pd
from app.configs import CatBoostConfig
from app.core.data_manager import DataManager
from app.features import create_time_features, update_extended_features_lastrow
from app.schemas import ExperimentMetrics
from app.trainers import BaseModelTrainer
from catboost import CatBoostRegressor

logger = logging.getLogger(__name__)


class CatBoostTrainer(BaseModelTrainer):
    config_class = CatBoostConfig

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.model = None
        self.df = None
        self.feature_columns = []
        self.target_column = "target"

    def _calculate_metrics(self, preds, reals) -> ExperimentMetrics:
        return ExperimentMetrics(
            mse=float(np.mean((preds - reals) ** 2)),
            mae=float(np.mean(np.abs(preds - reals))),
        )

    def train(self, ticker: str, base_date: pd.Timestamp, config: config_class):

        if self.data_manager is None:
            raise ValueError("DataManager не установлен.")

        logger.info(f"Начало обучения модели CatBoost по тикеру '{ticker}' до {base_date.date()}")

        try:
            df = self.data_manager.get_features(ticker, base_date)

            if df.empty or df[self.target_column].dropna().empty:
                raise ValueError(f"Недостаточно данных для обучения по тикеру '{ticker}'")

            self.df = df.copy()
            self.feature_columns = [col for col in df.columns if col != self.target_column]

            X = df[self.feature_columns]
            y = df[self.target_column]

            model_params = config.get_model_params()

            self.model = CatBoostRegressor(**model_params, verbose=0)
            self.model.fit(X, y)

            preds = self.model.predict(X)
            metrics = self._calculate_metrics(preds, y)

            logger.info(f"Обучение завершено. MSE={metrics.mse:.2f}, MAE={metrics.mae:.2f}")

            return self.model, metrics

        except Exception:
            logger.exception("Ошибка при обучении модели CatBoost")
            raise

    def predict(self, steps: int) -> tuple[list, None]:
        if self.model is None or self.df is None:
            raise ValueError("Сначала необходимо обучить модель")

        logger.info(f"Выполняется прогноз на {steps} шагов вперёд")

        predictions = []
        df = self.df.copy()

        try:
            for _ in range(steps):
                last_row = df.iloc[[-1]]
                features = last_row[self.feature_columns]

                pred_value = self.model.predict(features)[0]
                predictions.append(pred_value)

                # Добавляем новую дату
                next_index = df.index[-1] + pd.Timedelta(days=1)
                new_row = pd.DataFrame(index=[next_index], columns=df.columns)
                df = pd.concat([df, new_row])

                # Обновляем признаки в последней строке
                time_features = create_time_features(new_row, return_new_colnames=False)
                for col in time_features.columns:
                    df.loc[next_index, col] = time_features.loc[next_index, col]

                df = update_extended_features_lastrow(df, self.target_column, pred_value)

        except Exception:
            logger.exception("Ошибка при прогнозировании")
            raise

        logger.info("Прогноз успешно выполнен")
        return predictions, None
