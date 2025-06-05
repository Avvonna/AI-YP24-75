from abc import ABC, abstractmethod
from typing import Any


class BaseModelTrainer(ABC):
    @abstractmethod
    def train(self, data, config):
        pass

    @abstractmethod
    def predict(self, steps: int):
        pass

    @abstractmethod
    def predict_from_data(
        self, data: list[float], steps: int, config: Any
    ) -> tuple[list[float], dict[str, list[float]]]:
        pass
