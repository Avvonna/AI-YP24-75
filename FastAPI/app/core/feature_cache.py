import logging
import pickle
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FEATURES_PATH = DATA_DIR / "features.pkl"

logger = logging.getLogger(__name__)


class FeatureCache:
    def __init__(self, path: Path = FEATURES_PATH):
        self.path = path
        self.cache: dict[str, pd.DataFrame] = {}

    def get_all(self):
        return list(self.cache.keys())

    def get(self, ticker: str) -> pd.DataFrame:
        return self.cache.get(ticker)

    def add(self, ticker: str, features: pd.DataFrame):
        self.cache[ticker] = features
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as f:
            pickle.dump(self.cache, f)
        logger.info(f"Кэш признаков сохранён в {self.path}")

    def load(self):
        if self.path.exists():
            with self.path.open("rb") as f:
                self.cache = pickle.load(f)
            logger.info(f"Кэш признаков загружен из {self.path}")
