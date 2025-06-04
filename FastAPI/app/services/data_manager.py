import logging
import pickle
from pathlib import Path

import pandas as pd
from app.services.feature_engineering import preprocess_for_model

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INIT_DATA_PATH = DATA_DIR / "init_data.csv"
FEATURES_PATH = DATA_DIR / "features.pkl"

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self, tickers: list[str], csv_path: str | Path = INIT_DATA_PATH):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV-файл не найден: {self.csv_path.resolve()}")

        self.data = pd.read_csv(self.csv_path)
        self.data["date"] = pd.to_datetime(self.data["date"])

        self.tickers = tickers
        self.feature_cache = {}

        logger.info(f"Загружено {len(self.data)} строк данных из {self.csv_path.resolve()}")

    def get_ticker_history(self, ticker: str, start_date=None, end_date=None) -> dict:
        if ticker not in self.data.columns:
            raise ValueError(f"Тикер '{ticker}' отсутствует в исходных данных.")

        temp_df = self.data[["date", ticker]]

        if start_date:
            temp_df = temp_df[temp_df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            temp_df = temp_df[temp_df["date"] <= pd.to_datetime(end_date)]

        logger.debug(f"Возвращается история по тикеру {ticker}: {len(temp_df)} записей")
        return {
            "ticker": ticker,
            "dates": temp_df["date"].tolist(),
            "values": temp_df[ticker].tolist(),
        }

    def filter_data_for_training(self, ticker: str, base_date: pd.Timestamp, window: int = 60):
        min_date = self.data["date"].min()
        max_date = self.data["date"].max()

        original_base_date = base_date
        base_date = min(base_date, max_date)
        border = max(base_date - pd.Timedelta(days=window), min_date)

        df = self.data[(self.data["date"] >= border) & (self.data["date"] <= base_date)]

        if df.empty:
            logger.warning(
                f"Пустой DataFrame: тикер='{ticker}', окно={window} дней, "
                f"дата={original_base_date.date()}, обрезано до {base_date.date()}"
            )
            raise ValueError(
                f"Нет данных для тикера '{ticker}' в интервале {border.date()} — {base_date.date()}"
            )

        if original_base_date != base_date:
            logger.info(
                f"Дата base_date обрезана с {original_base_date.date()} до {base_date.date()} "
                f"(тикер: {ticker})"
            )

        logger.debug(f"Отобрано {len(df)} строк для обучения по тикеру {ticker}")
        return df[["date", ticker]]

    def generate_and_cache_features(self):
        for ticker in self.tickers:
            df = self.data[["date", ticker]].copy().dropna()
            df = df.rename(columns={ticker: "target"}).set_index("date").copy()
            processed = preprocess_for_model(df, target_column="target")
            self.feature_cache[ticker] = processed
            logger.info(f"Сгенерированы признаки для {ticker}: {processed.shape}")

    def save_features(self, path: str | Path = FEATURES_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            pickle.dump(self.feature_cache, f)

        logger.info(f"Сохранены признаки в {path.resolve()}")

    def load_features(self, path: str | Path = FEATURES_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл признаков не найден: {path.resolve()}")

        with path.open("rb") as f:
            self.feature_cache = pickle.load(f)

        logger.info(f"Загружены признаки из {path.resolve()}")

    def get_features(
        self, ticker: str,
        base_date: pd.Timestamp,
        force_recompute: bool = False
    ) -> pd.DataFrame:
        if ticker not in self.data.columns:
            raise ValueError(f"Тикер '{ticker}' отсутствует в исходных данных.")

        # Генерация признаков, если тикера нет в кэше или требуется обновление
        if force_recompute or ticker not in self.feature_cache:
            logger.info(f"{'Перегенерация' if force_recompute else 'Генерация'} признаков для тикера '{ticker}'")

            df = self.data[["date", ticker]].copy().dropna()
            if df.empty:
                raise ValueError(f"Нет данных по тикеру '{ticker}' для генерации признаков")

            df = df.rename(columns={ticker: "target"})
            df.set_index("date", inplace=True)

            processed = preprocess_for_model(df, target_column="target")
            self.feature_cache[ticker] = processed
            self.save_features()

            logger.info(f"Признаки по '{ticker}' сгенерированы и добавлены в кэш: {processed.shape}")

        df = self.feature_cache[ticker]
        df = df[df.index <= base_date]

        if df.empty:
            raise ValueError(f"Нет признаков по тикеру '{ticker}' до {base_date.date()}")

        return df
