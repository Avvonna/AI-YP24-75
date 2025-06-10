import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class CSVLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def list_tickers(self) -> list[str]:
        return [f.stem.upper() for f in self.data_dir.glob("*.csv")]

    def load_ticker(self, ticker: str) -> pd.DataFrame:
        path = self.data_dir / f"{ticker}.csv"
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df.sort_values("date", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Ошибка при загрузке {path}: {e}")
            return pd.DataFrame()

    def save_ticker(self, ticker: str, data: pd.DataFrame):
        path = self.data_dir / f"{ticker}.csv"
        data.to_csv(path, index=False)
        logger.info(f"Данные для {ticker} сохранены в {path}")

    def delete_ticker(self, ticker: str):
        path = self.data_dir / f"{ticker}.csv"
        if path.exists():
            path.unlink()
            logger.info(f"Удалён файл {path}")
