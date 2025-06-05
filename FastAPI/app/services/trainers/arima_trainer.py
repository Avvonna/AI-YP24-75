import logging

import numpy as np
import pandas as pd
from app.configs.arima_config import AutoARIMAConfig
from app.models.schemas import TimeSeriesData
from app.services.trainers.base_trainer import BaseModelTrainer
from pmdarima import auto_arima

logger = logging.getLogger(__name__)


class AutoARIMATrainer(BaseModelTrainer):
    def train(self, data: TimeSeriesData, config: AutoARIMAConfig):
        logger.info("Начало обучения модели AutoARIMA")

        try:
            df = pd.DataFrame({
                "date": pd.to_datetime(data.dates),
                "value": pd.to_numeric(data.values, errors="coerce")
            }).set_index("date")

            values = df["value"].values
            values = np.asarray(values, dtype=np.float64)

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
            logger.info(f"Обучение завершено. Метрики: AIC={metrics['aic']}, BIC={metrics['bic']}, "
                        f"MSE={metrics['mse']}, MAE={metrics['mae']}")
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

    def predict_from_data(self, data: list[float], steps: int, config: AutoARIMAConfig):
        logger.info(f"Начало прогноза из переданных данных (длина={len(data)}) на {steps} шагов")
        try:
            values = np.asarray(data, dtype=np.float64)
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
            logger.info("Обучение модели из новых данных завершено, запускается прогноз")
            return self.predict(steps)
        except Exception:
            logger.exception("Ошибка при прогнозе из новых данных")
            raise
