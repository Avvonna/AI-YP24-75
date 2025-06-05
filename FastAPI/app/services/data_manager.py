import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self, csv_path: str, tickers: list[str]):
        self.data = pd.read_csv(csv_path)
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.tickers = tickers
        logger.info(f"Загружено {len(self.data)} строк данных из {csv_path}")

    def get_ticker_history(self, ticker: str, start_date=None, end_date=None) -> dict:
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
