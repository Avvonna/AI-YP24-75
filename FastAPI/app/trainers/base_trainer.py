from abc import ABC, abstractmethod


class BaseModelTrainer(ABC):
    @abstractmethod
    def train(self, data, config):
        pass

    @abstractmethod
    def predict(self, steps: int):
        pass
