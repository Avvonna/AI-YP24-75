import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from app.models.data_models import ModelConfig, TimeSeriesData
from pmdarima import auto_arima

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


class MLService:
    def __init__(self):
        self.experiments: dict[str, dict] = {}
        self.current_experiment: Optional[str] = None
        self.current_model = "auto_arima_60"

    def train_model(self, data: TimeSeriesData, config: ModelConfig) -> dict[str, float]:
        """Обучение модели"""
        try:
            df = pd.DataFrame({
                "date": pd.to_datetime(data.dates),
                "value": pd.to_numeric(data.values, errors="coerce")
            }).set_index("date")

            if df["value"].isnull().any():
                raise ValueError("В данных обнаружены пропущенные значения")

            values = df["value"].values
            values = np.asarray(values, dtype=np.float64)

            if not np.all(np.isfinite(values)):
                raise ValueError("Обнаружены некорректные значения")

            model = auto_arima(
                values,
                seasonal=config.seasonal,
                m=config.seasonal_period if config.seasonal else 1,
                max_p=config.max_p,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                with_intercept=True
            )

            metrics = self._calculate_metrics(model, df["value"])

            self.experiments[data.experiment_name] = {
                "model": model,
                "training_history": data.values,
                "training_dates": data.dates,
                "config": config.dict(),
                "metrics": metrics
            }

            self.current_experiment = data.experiment_name
            return metrics

        except Exception as e:
            raise ValueError(f"Ошибка при обучении модели: {str(e)}") from e

    def predict(self, steps: int, experiment_name: Optional[str] = None) -> tuple[list[float], dict[str, list[float]]]:
        """Генерация прогноза с доверительными интервалами"""
        try:
            exp_name = experiment_name or self.current_experiment
            if not exp_name or exp_name not in self.experiments:
                raise ValueError("Эксперимент не найден")

            model = self.experiments[exp_name]["model"]
            forecast, conf_int = model.predict(
                n_periods=steps,
                return_conf_int=True,
                alpha=0.05
            )

            return (
                forecast.tolist(),
                {
                    "lower": conf_int[:, 0].tolist(),
                    "upper": conf_int[:, 1].tolist()
                }
            )
        except Exception as e:
            raise ValueError(f"Ошибка при прогнозировании: {str(e)}") from e

    def get_ticker_data(self, ticker: str, start_date: str, end_date: str) -> dict[str, Any]:
        """Получение данных тикера"""
        try:
            # Тестовые данные
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            dates = pd.date_range(start=start_date, end=end_date)

            # Генерируем тестовые значения
            trend = np.linspace(100, 120, len(dates))
            seasonality = 10 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
            noise = np.random.normal(0, 2, len(dates))
            values = trend + seasonality + noise

            return {
                "dates": dates.strftime("%Y-%m-%d").tolist(),
                "values": values.tolist()
            }
        except Exception as e:
            raise ValueError(f"Ошибка при получении данных: {str(e)}") from e

    def set_model(self, model_name: str) -> dict[str, str]:
        """Выбор модели прогнозирования"""
        if model_name != "auto_arima_60":
            raise ValueError("Неподдерживаемая модель. Доступна только auto_arima_60")
        self.current_model = model_name
        return {"status": "успешно"}

    def train_and_predict(self, ticker: str, base_date: str, forecast_period: int = 10) -> dict[str, Any]:
        """Обучение модели и прогнозирование"""
        try:
            # Получаем исторические данные
            historical_data = self.get_ticker_data(
                ticker,
                pd.to_datetime(base_date) - pd.Timedelta(days=60),
                base_date
            )

            # Подготавливаем данные для обучения
            train_data = TimeSeriesData(
                dates=[pd.to_datetime(d) for d in historical_data["dates"]],
                values=historical_data["values"],
                experiment_name=f"{ticker}_{base_date}"
            )

            # Обучаем модель
            self.train_model(train_data, ModelConfig())

            # Делаем прогноз
            forecast, conf_intervals = self.predict(forecast_period)

            # Создаем даты для прогноза
            forecast_dates = pd.date_range(
                start=pd.to_datetime(base_date),
                periods=forecast_period+1
            )[1:]

            return {
                "forecast_dates": forecast_dates.strftime("%Y-%m-%d").tolist(),
                "forecast_values": forecast,
                "confidence_intervals": conf_intervals,
                "history": {
                    "dates": historical_data["dates"],
                    "values": historical_data["values"]
                }
            }
        except Exception as e:
            raise ValueError(f"Ошибка при прогнозировании: {str(e)}") from e

    def compare_experiments(self, experiment_names: set[str]) -> dict[str, dict]:
        """Сравнение метрик и результатов экспериментов"""
        try:
            if not experiment_names:
                raise ValueError("Не указаны эксперименты для сравнения")

            available = set(self.experiments.keys())
            to_compare = experiment_names.intersection(available)
            missing = experiment_names - available

            results = {}

            if to_compare:
                results["experiments"] = self._prepare_comparison_results(to_compare)

            if missing:
                results["missing_experiments"] = {
                    "count": len(missing),
                    "names": list(missing)
                }

            if not to_compare:
                results["status"] = "warning"
                results["message"] = "Нет доступных экспериментов для сравнения"

            return results

        except Exception as e:
            raise ValueError(f"Ошибка при сравнении экспериментов: {str(e)}") from e

    def _prepare_comparison_results(self, experiments: set[str]) -> dict[str, dict]:
        """Подготовка результатов для сравнения"""
        results = {}
        for name in experiments:
            exp = self.experiments[name]
            results[name] = {
                "metrics": exp["metrics"],
                "config": exp["config"],
                "training_history": {
                    "dates": [d.isoformat() if isinstance(d, datetime) else d
                             for d in exp["training_dates"]],
                    "values": exp["training_history"]
                }
            }
        return results

    def _calculate_metrics(self, model: auto_arima, data: pd.Series) -> dict[str, float]:
        """Расчет метрик качества модели"""
        try:
            predictions = model.predict_in_sample()
            mse = float(np.mean((data.values - predictions) ** 2))
            mae = float(np.mean(np.abs(data.values - predictions)))

            return {
                "aic": float(model.aic()),
                "bic": float(model.bic()),
                "mse": mse,
                "mae": mae
            }
        except Exception as e:
            raise ValueError(f"Ошибка при расчете метрик: {str(e)}") from e

    def save_model(self, path: str, experiment_name: Optional[str] = None, format_type: str = "pickle") -> None:
        """Сохранение модели в файл"""
        try:
            exp_name = experiment_name or self.current_experiment
            if not exp_name or exp_name not in self.experiments:
                raise ValueError("Эксперимент не найден")

            path = Path(path)
            if format_type == "pickle":
                with open(path, "wb") as f:
                    pickle.dump(self.experiments[exp_name], f)
            elif format_type == "joblib":
                joblib.dump(self.experiments[exp_name], path)
            else:
                raise ValueError("Неподдерживаемый формат")
        except Exception as e:
            raise ValueError(f"Ошибка при сохранении модели: {str(e)}") from e

    def load_model(self, path: str, experiment_name: str, format_type: str = "pickle") -> None:
        """Загрузка модели из файла"""
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Файл модели не найден: {path}")

            if format_type == "pickle":
                with open(path, "rb") as f:
                    experiment_data = pickle.load(f)
            elif format_type == "joblib":
                experiment_data = joblib.load(path)
            else:
                raise ValueError("Неподдерживаемый формат")

            self.experiments[experiment_name] = experiment_data
            self.current_experiment = experiment_name

        except Exception as e:
            raise ValueError(f"Ошибка при загрузке модели: {str(e)}") from e
