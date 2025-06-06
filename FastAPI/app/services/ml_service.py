from datetime import datetime
import pandas as pd
from app.configs.arima_config import AutoARIMAConfig
from app.configs.catboost_config import CatBoostConfig
from app.models.schemas import TimeSeriesData
from app.services.data_manager import DataManager
from app.services.experiment_manager import ExperimentManager
from app.services.model_registry import ModelRegistry


class MLService:
    """В классе реализованы методы для работы с моделями"""
    def __init__(self):
        """
        Инициализирует MLService

        Atributes:
        data_manger: Методы для работы с данными
        tickers (list): Список доступных тикеров
        registry: Доступные модели
        experiments: Методы для работы с результатами экспериментов
        current_model: Текущая модель
        """
        self.data_manager = DataManager()
        self.tickers = self.data_manager.tickers
        self.registry = ModelRegistry()
        self.experiments = ExperimentManager()
        self.current_model = "auto_arima"

    def set_model(self, model_name: str):
        """
        Устанавливает модель
        Args:
            model_name: Название модели
        """
        if model_name not in self.registry.models:
            raise ValueError(f"Модель '{model_name}' не поддерживается.")
        self.current_model = model_name
        return {"status": "модель выбрана", "model": model_name}

    def get_available_tickers(self):
        """Возвращает список доступных тикеров"""
        return self.tickers

    def add_ticker_data(self, ticker: str, data: pd.DataFrame):
        """
        Добавляет данные нового тикера или
        обновляет данные уже имеющегося тикера
        Args:
            ticker: Название добавляемого тикера
            data: DataFrame с данными добавляемого тикера
        """
        return self.data_manager.add_ticker_data(ticker, data)

    def remove_ticker(self, ticker: str):
        """
        Удаляет тикер из списка доступных тикеров,
        а так же удаляет данные тикера
        Args:
            ticker: Название удаляемого тикера
        """
        return self.data_manager.remove_ticker(ticker)

    def get_ticker_history(self, ticker, start_date, end_date):
        """
        Возвращает отфильтрованные по дате данные тикера,
        если временные рамки не указаны, то возвращает все данные
        Args:
            ticker: Название тикера
            start_date: Дата начала периода
            end_date: Дата окончания периода
        """
        return self.data_manager.get_ticker_history(
            ticker,
            start_date,
            end_date)

    def train_and_predict(self, ticker: str, base_date: str, forecast_period: int, config):
        base_dt = datetime.fromisoformat(base_date)
        df = self.data_manager.filter_data_for_training(ticker, base_dt)
        dates = df["date"].tolist()
        values = df[ticker].tolist()

        ts_data = TimeSeriesData(
            dates=dates,
            values=values,
            experiment_name=f"{ticker}_{base_date}_{config.model_type}"
        )

        model_type = getattr(config, "model_type", self.current_model)
        trainer = self.registry.get_trainer(model_type)

        if model_type == "auto_arima":
            assert isinstance(config, AutoARIMAConfig), "Конфигурация должна быть AutoARIMAConfig"
        elif model_type == "catboost":
            assert isinstance(config, CatBoostConfig), "Конфигурация должна быть CatBoostConfig"
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")

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
        model_type = getattr(config, "model_type", self.current_model)
        trainer = self.registry.get_trainer(model_type)
        return trainer.predict_from_data(data, steps, config)

    def compare_experiments(self, names: list[str]):
        return self.experiments.compare(names)


ml_service = MLService()
