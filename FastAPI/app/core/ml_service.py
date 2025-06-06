from datetime import datetime

import pandas as pd
from app.configs import AutoARIMAConfig, CatBoostConfig, LSTMConfig
from app.core import DataManager, ExperimentManager, ModelRegistry
from app.schemas import TimeSeriesData


class MLService:
    def __init__(self):
        self.tickers = ["LKOH", "ROSN", "SIBN", "SNGS", "TATN"]
        self.data_manager = DataManager(self.tickers)
        self.registry = ModelRegistry(self.data_manager)
        self.experiments = ExperimentManager()
        self.current_model = "auto_arima"

    def set_model(self, model_name: str):
        if model_name not in self.registry.models:
            raise ValueError(f"Модель '{model_name}' не поддерживается.")
        self.current_model = model_name
        return {"status": "модель выбрана", "model": model_name}

    def get_available_tickers(self):
        return self.tickers

    def add_ticker(self, ticker: str):
        if ticker not in self.tickers:
            self.tickers.append(ticker)

    def remove_ticker(self, ticker: str):
        if ticker in self.tickers:
            self.tickers.remove(ticker)

    def get_ticker_history(self, ticker, start_date, end_date):
        return self.data_manager.get_ticker_history(ticker, start_date, end_date)

    def train_and_predict(self, ticker: str, base_date: str, forecast_period: int, config):
        base_dt = datetime.fromisoformat(base_date)
        model_type = getattr(config, "model_type", self.current_model)
        trainer = self.registry.get_trainer(model_type)

        if model_type == "auto_arima":
            assert isinstance(config, AutoARIMAConfig)
        elif model_type == "catboost":
            assert isinstance(config, CatBoostConfig)
        elif model_type == "lstm":
            assert isinstance(config, LSTMConfig)
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")

        model, metrics = trainer.train(ticker=ticker, base_date=base_dt, config=config)
        forecast, conf = trainer.predict(forecast_period)

        history_dict = self.data_manager.get_ticker_history(ticker, None, base_dt)
        dates = history_dict["dates"]
        values = history_dict["values"]

        ts_data = TimeSeriesData(
            dates=dates,
            values=values,
            experiment_name=f"{ticker}_{base_date}_{model_type}"
        )

        self.experiments.save(ts_data.experiment_name, model, config, metrics, {"dates": dates, "values": values})

        forecast_dates = pd.date_range(base_dt, periods=forecast_period + 1)[1:]

        return {
            "forecast_dates": forecast_dates.tolist(),
            "forecast_values": forecast,
            "confidence_intervals": conf,
            "history": {"dates": dates, "values": values}
        }

    def compare_experiments(self, names: list[str]):
        return self.experiments.compare(names)

ml_pipeline = MLService()
