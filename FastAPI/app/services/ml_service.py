from datetime import datetime
from pathlib import Path

import pandas as pd
from app.models.schemas import TimeSeriesData
from app.services.data_manager import DataManager
from app.services.experiment_manager import ExperimentManager
from app.services.model_registry import ModelRegistry

DATA_PATH = Path(__file__).resolve().parent / "init_data.csv"

class MLService:
    def __init__(self):
        self.tickers = ["LKOH", "ROSN", "SIBN", "SNGS", "TATN"]
        self.data_manager = DataManager(str(DATA_PATH), self.tickers)
        self.registry = ModelRegistry()
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
        df = self.data_manager.filter_data_for_training(ticker, base_dt)
        dates = df["date"].tolist()
        values = df[ticker].tolist()

        ts_data = TimeSeriesData(
            dates=dates,
            values=values,
            experiment_name=f"{ticker}_{base_date}_{self.current_model}"
        )

        trainer = self.registry.get_trainer(self.current_model)
        model, metrics = trainer.train(ts_data, config)

        forecast, conf = trainer.predict(forecast_period)

        self.experiments.save(ts_data.experiment_name, model, config, metrics, {"dates": dates, "values": values})

        forecast_dates = pd.date_range(base_dt, periods=forecast_period + 1)[1:]

        return {
            "forecast_dates": forecast_dates.tolist(),
            "forecast_values": forecast,
            "confidence_intervals": conf,
            "history": {"dates": dates, "values": values}
        }

    def predict_current_model(self, data: list[float], steps: int, config):
        trainer = self.registry.get_trainer(self.current_model)
        return trainer.predict_from_data(data, steps, config)

    def compare_experiments(self, names: list[str]):
        return self.experiments.compare(names)

ml_service = MLService()
