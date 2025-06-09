import logging
from datetime import date
from pathlib import Path

import pandas as pd
from app.core.csv_loader import CSVLoader
from app.core.feature_cache import FeatureCache
from app.features import preprocess_for_model
from app.schemas import TickerHistory
from app.utils import validate_dataframe
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INIT_DATA_PATH = DATA_DIR / "init_data.csv"
FEATURES_PATH = DATA_DIR / "features.pkl"

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self, csv_path: Path = DATA_DIR):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Каталог не найден: {self.csv_path.resolve()}")

        self.loader = CSVLoader(self.csv_path)
        self.feature_cache = FeatureCache()
        self.tickers = self.loader.list_tickers()
        self.data = {ticker: self.loader.load_ticker(ticker) for ticker in self.tickers}

    def get_available_tickers(self) -> list[str]:
        """Возвращает список доступных тикеров"""
        return self.tickers

    def add_ticker_data(self, ticker: str, new_data: pd.DataFrame):
        """
        Добавляет данные нового тикера или
        обновляет данные уже имеющегося тикера
        Args:
            ticker: Название добавляемого тикера
            data: DataFrame с данными добавляемого тикера
        """
        ticker = ticker.upper()
        new_data.columns = [col.upper() if col != "date" else col for col in new_data.columns]
        validated_df = validate_dataframe(new_data, ticker)

        current_df = self.data.get(ticker, pd.DataFrame())
        updated_df = pd.concat([current_df, validated_df]).drop_duplicates("date").sort_values("date")
        self.data[ticker] = updated_df

        if ticker not in self.tickers:
            self.tickers.append(ticker)

        self.loader.save_ticker(ticker, updated_df)

    def remove_ticker(self, ticker: str):
        """
        Удаляет тикер из списка доступных тикеров,
        а так же удаляет данные тикера
        Args:
            ticker: Название удаляемого тикера
        """
        ticker = ticker.upper()
        if ticker not in self.tickers:
            logger.warning(f"Тикер {ticker} не найден")
            return

        self.loader.delete_ticker(ticker)
        self.tickers.remove(ticker)
        del self.data[ticker]

    def get_ticker_history(
        self,
        ticker: str,
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp]
    ) -> TickerHistory:
        """
        Возвращает отфильтрованные по дате данные тикера,
        если временные рамки не указаны, то возвращает все данные
        Args:
            ticker: Название тикера
            start_date: Дата начала периода
            end_date: Дата окончания периода
        """
        ticker = ticker.upper()
        if ticker not in self.data:
            raise ValueError(f"Тикер '{ticker}' не найден")

        df = self.data[ticker].copy()
        df = df.dropna(subset=[ticker])
        start = df["date"].min() if pd.isna(start_date) else start_date
        end = df["date"].max() if pd.isna(end_date) else end_date

        mask = (df["date"] >= start) & (df["date"] <= end)
        df = df.loc[mask]

        return TickerHistory(
            ticker=ticker,
            dates=df["date"].dt.strftime("%Y-%m-%d").tolist(),
            values=df[ticker].tolist()
        )

    def filter_data_for_training(self, ticker: str, base_date: date, window: int = 60):
        """
        Возвращает заданное количество (по умолчанию - 60) timestamps
        до заданной даты
        Args:
            ticker: Название тикера
            base_date: Конечная дата
            window: Количество timestamps до конечной даты
        """
        df = self.data[ticker]

        base_date = min(pd.to_datetime(base_date), df["date"].max())
        border = max(base_date - pd.Timedelta(days=window), df["date"].min())

        filtered = df[(df["date"] >= border) & (df["date"] <= base_date)]
        if filtered.empty:
            raise ValueError(f"Нет данных для '{ticker}' в интервале {border.date()} — {base_date.date()}")
        return filtered[["date", ticker]]

    def get_features(self, ticker: str, base_date: pd.Timestamp, force_recompute: bool = False) -> pd.DataFrame:
        if ticker not in self.data:
            raise ValueError(f"Тикер '{ticker}' отсутствует в данных")

        if force_recompute or ticker not in self.feature_cache.get_all():
            df = self.data[ticker][["date", ticker]].dropna()
            df = df.rename(columns={ticker: "target"}).set_index("date")
            features = preprocess_for_model(df, target_column="target")
            self.feature_cache.add(ticker, features)

        df = self.feature_cache.get(ticker)
        df = df[df.index <= base_date]

        if df.empty:
            raise ValueError(f"Нет признаков по '{ticker}' до {base_date.date()}")
        return df
