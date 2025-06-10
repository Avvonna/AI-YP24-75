from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd
from app.configs import BaseModelConfig
from app.schemas import ExperimentMetrics, ForecastConfidenceIntervals


class BaseModelTrainer(ABC):
    config_class: type[BaseModelConfig]

    def get_config_schema(self) -> dict:
        return self.config_class.model_json_schema()

    @abstractmethod
    def train(self, ticker: str, base_date: pd.Timestamp, config: BaseModelConfig) -> tuple[Any, ExperimentMetrics]:
        pass

    @abstractmethod
    def predict(self, steps: int) -> tuple[list, Optional[ForecastConfidenceIntervals]]:
        pass
