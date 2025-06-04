import numpy as np
from app.models.schemas import ModelConfig
from pmdarima import auto_arima


class ModelPredictor:
    def predict(self, model, steps: int):
        forecast, conf_int = model.predict(n_periods=steps, return_conf_int=True, alpha=0.05)
        return forecast.tolist(), {
            "lower": conf_int[:, 0].tolist(),
            "upper": conf_int[:, 1].tolist()
        }

    def predict_from_data(self, data: list[float], steps: int, config: ModelConfig):
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

        return self.predict(model, steps)
