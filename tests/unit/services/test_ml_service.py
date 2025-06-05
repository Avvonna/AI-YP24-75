from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from app.configs.arima_config import AutoARIMAConfig
from app.configs.catboost_config import CatBoostConfig
from app.services.ml_service import MLService


@pytest.fixture
def ml_service():
    return MLService()

@pytest.fixture
def fake_data():
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "LKOH": [100, 101, 102]
    })

@pytest.mark.parametrize("model_type,config_class", [
    ("auto_arima", AutoARIMAConfig),
    ("catboost", CatBoostConfig),
])
@patch("app.services.ml_service.ModelRegistry.get_trainer")
@patch("app.services.ml_service.DataManager.filter_data_for_training")
@patch("app.services.ml_service.ExperimentManager.save")
def test_train_and_predict_mocked_models(
    mock_save, mock_filter, mock_get_trainer,
    ml_service, fake_data, model_type, config_class
):
    mock_filter.return_value = fake_data

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = ("fake_model", {"mae": 0.1, "mse": 0.2})
    mock_trainer.predict.return_value = (
        np.array([105.0, 106.0]),
        {
            "lower": np.array([104.0, 105.0]),
            "upper": np.array([106.0, 107.0])
        }
    )
    mock_get_trainer.return_value = mock_trainer

    config = config_class(model_type=model_type)

    result = ml_service.train_and_predict(
        ticker="LKOH",
        base_date="2024-01-03",
        forecast_period=2,
        config=config
    )

    assert list(result["forecast_values"]) == [105.0, 106.0]
    assert list(result["confidence_intervals"]["lower"]) == [104.0, 105.0]
    assert list(result["confidence_intervals"]["upper"]) == [106.0, 107.0]
    assert "history" in result

    mock_trainer.train.assert_called_once()
    mock_trainer.predict.assert_called_once()
    mock_save.assert_called_once()
