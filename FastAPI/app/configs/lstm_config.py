from typing import Literal

from app.configs.base_config import BaseModelConfig
from pydantic import Field


class LSTMConfig(BaseModelConfig):
    model_type: Literal["lstm"] = "lstm"
    hidden_dim: int = Field(default=64)
    num_layers: int = Field(default=2)
    dropout: float = Field(default=0.2)
    lr: float = Field(default=0.001)
    epochs: int = Field(default=100)
    batch_size: int = Field(default=32)
    patience: int = Field(default=10)
    window_size: int = Field(default=30, description="Размер временного окна для входа LSTM")
