import numpy as np
import pandas as pd
from app.models.base_model import BaseModelTrainer
from app.models.schemas import ModelConfig, TimeSeriesData
from pmdarima import auto_arima


class AutoARIMATrainer(BaseModelTrainer):
    def train(self, data: TimeSeriesData, config: ModelConfig):
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

        return model, self._calculate_metrics(model, df["value"])

    def _calculate_metrics(self, model, data: pd.Series):
        preds = model.predict_in_sample()
        return {
            "aic": float(model.aic()),
            "bic": float(model.bic()),
            "mse": float(np.mean((data.values - preds) ** 2)),
            "mae": float(np.mean(np.abs(data.values - preds)))
        }
