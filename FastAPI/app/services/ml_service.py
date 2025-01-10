import logging
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from app.models.data_models import ModelConfig, TimeSeriesData
from pmdarima import auto_arima

logger = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)

init_data = pd.read_csv("app/services/init_data.csv")
init_data["date"] = pd.to_datetime(init_data["date"])

init_tickers = ["LKOH", "ROSN", "SIBN", "SNGS", "TATN"]

class MLService:
    def __init__(self):
        self.experiments: dict[str, dict] = {}
        self.current_experiment: Optional[str] = None
        self.current_model = "auto_arima_60"
        self.tickers: list[str] = init_tickers
        self.data: pd.DataFrame = init_data
        logger.info("MLService инициализирован")

    def get_available_tickers(self) -> list[str]:
        return self.tickers

    def add_ticker(self, ticker: str) -> None:
        if ticker not in self.tickers:
            self.tickers.append(ticker)
            logger.info(f"Добавлен тикер: {ticker}")

    def remove_ticker(self, ticker: str) -> None:
        if ticker in self.tickers:
            self.tickers.remove(ticker)
            logger.info(f"Удален тикер: {ticker}")

    def get_ticker_history(self, ticker, start_date, end_date) -> dict[str, Any]:
        try:
            temp_df = self.data[["date", ticker]]
            if start_date:
                temp_df = temp_df[temp_df["date"] >= start_date]
            if end_date:
                temp_df = temp_df[temp_df["date"] <= end_date]

            return {
                "ticker": ticker,
                "dates": temp_df["date"].tolist(),
                "values": temp_df[ticker].tolist(),
            }
        except Exception as e:
            raise Exception(e) from e

    def train_model(self, data: TimeSeriesData, config: ModelConfig) -> dict[str, float]:
        try:
            logger.info(f"Начало обучения модели для эксперимента {data.experiment_name}")
            df = pd.DataFrame({
                "date": pd.to_datetime(data.dates),
                "value": pd.to_numeric(data.values, errors="coerce")
            }).set_index("date")

            if df["value"].isnull().any():
                logger.error("Обнаружены пропущенные значения в данных")
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
            logger.info(f"Модель успешно обучена. Метрики: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {str(e)}")
            raise ValueError(f"Ошибка при обучении модели: {str(e)}") from e

    def predict(self, steps: int, experiment_name: Optional[str] = None) -> tuple[list[float], dict[str, list[float]]]:
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
            logger.error(f"Ошибка при прогнозировании: {str(e)}")
            raise ValueError(f"Ошибка при прогнозировании: {str(e)}") from e

    def predict_current_model(self, data: list[float], steps: int) -> tuple[list[float], dict[str, list[float]]]:
        try:
            logger.info(f"Прогнозирование current_model на {steps} шагов")

            if self.current_model != "auto_arima_60":
                logger.error(f"Неподдерживаемая модель: {self.current_model}")
                raise ValueError("Неподдерживаемая модель")

            values = np.asarray(data, dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError("Обнаружены некорректные значения")

            model = auto_arima(
                values,
                seasonal=True,
                m=7,
                max_p=5,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True
            )

            forecast, conf_int = model.predict(
                n_periods=steps,
                return_conf_int=True,
                alpha=0.05
            )

            logger.info("Прогноз успешно получен")
            return (
                forecast.tolist(),
                {
                    "lower": conf_int[:, 0].tolist(),
                    "upper": conf_int[:, 1].tolist()
                }
            )
        except Exception as e:
            logger.error(f"Ошибка предсказания current_model: {str(e)}")
            raise ValueError(f"Ошибка при прогнозировании: {str(e)}") from e

    def set_model(self, model_name: str) -> dict[str, str]:
        try:
            logger.info(f"Установка модели {model_name}")
            if model_name != "auto_arima_60":
                raise ValueError("Неподдерживаемая модель. Доступна только auto_arima_60")
            self.current_model = model_name
            return {"status": "успешно"}
        except Exception as e:
            logger.error(f"Ошибка установки модели: {str(e)}")
            raise ValueError(str(e)) from e

    def train_and_predict(self, ticker: str, base_date: str, forecast_period: int = 10) -> dict[str, Any]:
        try:
            logger.info(f"Обучение и прогнозирование для тикера {ticker}")

            if ticker not in self.tickers:
                raise ValueError(f"Тикер {ticker} не найден в списке доступных")
            base_date = datetime.fromisoformat(base_date)
            border = base_date - timedelta(days=60)
            filtered_data = self.data[(self.data["date"] >= border) & (self.data["date"] <= base_date)]

            dates=filtered_data["date"].tolist()
            values=filtered_data[ticker].tolist()

            train_data = TimeSeriesData(
                dates=dates,
                values=values,
                experiment_name=f"{ticker}_{base_date}"
            )

            self.train_model(train_data, ModelConfig())
            forecast, conf_intervals = self.predict(forecast_period)

            forecast_dates = pd.date_range(
                start=pd.to_datetime(base_date),
                periods=forecast_period+1
            )[1:]

            return {
                # "forecast_dates": forecast_dates.strftime("%Y-%m-%d").tolist(),
                "forecast_dates": forecast_dates.tolist(),
                "forecast_values": forecast,
                "confidence_intervals": conf_intervals,
                "history": {
                    # "dates": dates.strftime("%Y-%m-%d").tolist(),
                    "dates": dates,
                    "values": values
                }
            }
        except Exception as e:
            logger.error(f"Ошибка при прогнозировании: {str(e)}")
            raise ValueError(f"Ошибка при прогнозировании: {str(e)}") from e

    def _calculate_metrics(self, model: auto_arima, data: pd.Series) -> dict[str, float]:
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
            logger.error(f"Ошибка при расчете метрик: {str(e)}")
            raise ValueError(f"Ошибка при расчете метрик: {str(e)}") from e

    def compare_experiments(self, experiment_names: list[str]) -> dict[str, dict]:
        try:
            logger.info(f"Сравнение экспериментов: {experiment_names}")
            if not experiment_names:
                raise ValueError("Не указаны эксперименты для сравнения")

            results = {}
            available = set(self.experiments.keys())
            to_compare = set(experiment_names).intersection(available)
            missing = set(experiment_names) - available

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
            logger.error(f"Ошибка при сравнении экспериментов: {str(e)}")
            raise ValueError(f"Ошибка при сравнении экспериментов: {str(e)}") from e

    def _prepare_comparison_results(self, experiments: set) -> dict[str, dict]:
        try:
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

        except Exception as e:
            logger.error(f"Ошибка при подготовке результатов сравнения: {str(e)}")
            raise ValueError(f"Ошибка при подготовке результатов сравнения: {str(e)}") from e

    def save_model(self, path: str, experiment_name: Optional[str] = None, format_type: str = "pickle") -> None:
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

            logger.info(f"Модель успешно сохранена: {path}")

        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {str(e)}")
            raise ValueError(f"Ошибка при сохранении модели: {str(e)}") from e

    def load_model(self, path: str, experiment_name: str, format_type: str = "pickle") -> None:
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

            logger.info(f"Модель успешно загружена: {path}")

        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise ValueError(f"Ошибка при загрузке модели: {str(e)}") from e
