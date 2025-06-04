import os
import random
import sys

import pytest
from httpx import ASGITransport, AsyncClient

from fastapi import FastAPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../FastAPI")))

from app.api.predict import router as predict_router
from app.configs.arima_config import AutoARIMAConfig

random.seed(42)

app = FastAPI()
app.include_router(predict_router)

@pytest.mark.asyncio
async def test_predict_valid_request():
    request_data = {
        "ticker": "LKOH",
        "base_date": "2005-01-11",
        "forecast_period": 2,
        "config": AutoARIMAConfig().model_dump()
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/predict/", json=request_data)

    # print("Response JSON:", response.json())
    assert response.status_code == 200
    assert "forecast_dates" in response.json()


@pytest.mark.asyncio
async def test_predict_current_model_valid():
    request_data = {
        "data": [100 + random.uniform(-5, 5) for _ in range(60)],
        "steps": 2,
        "config": AutoARIMAConfig().model_dump()
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/predict/current", json=request_data)

    # print("Response JSON:", response.json())
    assert response.status_code == 200
    assert "forecast" in response.json()
