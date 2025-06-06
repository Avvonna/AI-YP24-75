import logging

import numpy as np
import pandas as pd
from app.configs import AutoARIMAConfig
from app.core import DataManager
from pmdarima import auto_arima

from .base_trainer import BaseModelTrainer

logger = logging.getLogger(__name__)


class AutoARIMATrainer(BaseModelTrainer):
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def train(self, ticker: str, base_date: pd.Timestamp, config: AutoARIMAConfig):
        logger.info("Начало обучения модели AutoARIMA")

        try:
            df_raw = self.data_manager.filter_data_for_training(ticker, base_date, window=60)
            df = pd.DataFrame({
                "date": pd.to_datetime(df_raw["date"]),
                "value": pd.to_numeric(df_raw[ticker], errors="coerce")
            }).dropna().set_index("date")

            values = df["value"].values.astype(np.float64)

            model = auto_arima(
                values,
                seasonal=config.seasonal,
                m=config.seasonal_period if config.seasonal else 1,
                max_p=config.max_p,
                max_d=config.max_d,
                max_q=config.max_q,
                max_P=config.max_P,
                max_D=config.max_D,
                max_Q=config.max_Q,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )

            self.model = model
            metrics = self._calculate_metrics(model, df["value"])

            logger.info(f"Обучение завершено. Метрики: AIC={metrics['aic']:.2f}, BIC={metrics['bic']:.2f}, "
                        f"MSE={metrics['mse']:.2f}, MAE={metrics['mae']:.2f}")
            return model, metrics

        except Exception:
            logger.exception("Ошибка при обучении модели AutoARIMA")
            raise

    def _calculate_metrics(self, model, data: pd.Series):
        preds = model.predict_in_sample()
        return {
            "aic": float(model.aic()),
            "bic": float(model.bic()),
            "mse": float(np.mean((data.values - preds) ** 2)),
            "mae": float(np.mean(np.abs(data.values - preds)))
        }

    def predict(self, steps: int):
        logger.info(f"Выполняется прогноз на {steps} шагов вперёд")
        try:
            forecast, conf_int = self.model.predict(n_periods=steps, return_conf_int=True)
            logger.info("Прогноз успешно выполнен")
            return forecast.tolist(), {
                "lower": conf_int[:, 0].tolist(),
                "upper": conf_int[:, 1].tolist()
            }
        except Exception:
            logger.exception("Ошибка при выполнении прогноза")
            raise
