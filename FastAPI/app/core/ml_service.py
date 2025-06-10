import logging
from datetime import date
from typing import Optional

import pandas as pd
from app.configs import ModelConfigUnion
from app.core.data_manager import DataManager
from app.core.experiment_manager import ExperimentManager
from app.core.model_registry import ModelRegistry
from app.schemas import ForecastResult, ModelSelectResponseSchema

logger = logging.getLogger(__name__)


class MLService:
    """В классе реализованы методы для работы с моделями"""
    def __init__(self):
        self.data_manager = DataManager()
        self.tickers = self.data_manager.tickers
        self.registry = ModelRegistry(self.data_manager)
        self.experiments = ExperimentManager()
        self.current_model = "auto_arima"

    def get_available_models(self) -> list[str]:
        return self.registry.get_available_models()

    def get_config_class(self, model_name: str):
        return self.registry.get_config(model_name)

    def get_config_schema(self, model_name: str):
        config_cls = self.get_config_class(model_name)
        return config_cls.model_json_schema()

    def set_model(self, model_name: str):
        if model_name not in self.registry.models:
            raise ValueError(f"Модель '{model_name}' не поддерживается.")
        self.current_model = model_name
        return ModelSelectResponseSchema(status="Модель выбрана", model_name=self.current_model)

    def get_available_tickers(self):
        return self.tickers

    def add_ticker_data(self, ticker: str, data: pd.DataFrame):
        return self.data_manager.add_ticker_data(ticker, data)

    def remove_ticker(self, ticker: str):
        return self.data_manager.remove_ticker(ticker)

    def get_ticker_history(
        self,
        ticker: str,
        start_date: Optional[date],
        end_date: Optional[date]
    ):
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        return self.data_manager.get_ticker_history(ticker, start_date, end_date)

    def get_trainer(self, model_name: str):
        return self.registry.get_trainer(model_name)

    def train_and_predict(
        self,
        ticker: str,
        base_date: date,
        forecast_period: int,
        config: ModelConfigUnion
    ) -> ForecastResult:
        base_date = pd.Timestamp(base_date)
        exp_name = f"{ticker}_{base_date.strftime('%Y-%m-%d')}_{config.model_type}"

        trainer = self.get_trainer(config.model_type)
        model, metrics = trainer.train(ticker=ticker, base_date=base_date, config=config)
        forecast, conf = trainer.predict(forecast_period)

        ticket_history = self.get_ticker_history(ticker, None, base_date)

        self.experiments.save(
            name=exp_name,
            model=model,
            config=config,
            metrics=metrics,
            history=ticket_history
        )

        forecast_dates = pd.date_range(base_date, periods=forecast_period + 1)[1:]

        return ForecastResult(
            experiment_id=exp_name,
            forecast_dates=forecast_dates.date.tolist(),
            forecast_values=forecast,
            confidence_intervals=conf,
        )

    def get_experiments_list(self):
        return self.experiments.get_all()

    def get_experiment(self, name: str):
        return self.experiments.get(name)

    def compare_experiments(self, names: list[str]):
        return self.experiments.compare(names)

ml_pipeline = MLService()
