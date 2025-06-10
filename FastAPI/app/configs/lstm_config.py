from pydantic import Field

from .base_config import BaseModelConfig


class LSTMConfig(BaseModelConfig):
    model_type: str = "lstm"

    hidden_dim: int = Field(
        default=64,
        description="Размер скрытого слоя LSTM (количество нейронов)"
    )

    num_layers: int = Field(
        default=1,
        description="Количество слоёв LSTM в модели"
    )

    dropout: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Доля dropout между слоями (от 0 до 1)"
    )

    lr: float = Field(
        default=0.001,
        description="Скорость обучения (learning rate) оптимизатора"
    )

    epochs: int = Field(
        default=10,
        description="Максимальное количество эпох обучения"
    )

    batch_size: int = Field(
        default=32,
        description="Размер батча для обучения модели"
    )

    patience: int = Field(
        default=5,
        description="Количество эпох без улучшения до ранней остановки (early stopping)"
    )

    window_size: int = Field(
        default=30,
        description="Размер временного окна для формирования входной последовательности"
    )

    save_model: bool = Field(
        default=True,
        description="Сохранять ли обученную модель в файл"
    )
