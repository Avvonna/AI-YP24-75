import os
import sys

import pytest
from httpx import ASGITransport, AsyncClient

from fastapi import FastAPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../FastAPI")))

from app.api.experiments import router as experiments_router

app = FastAPI()
app.include_router(experiments_router)

class FakeMLService:
    def compare_experiments(self, names):
        if not names:
            raise ValueError("Список имён экспериментов пуст.")
        return {
            "comparison": f"Сравнение {', '.join(names)} успешно завершено"
        }

@pytest.mark.asyncio
async def test_experiments_compare_success(monkeypatch):
    fake_service = FakeMLService()
    monkeypatch.setattr("app.api.experiments.ml_service", fake_service)

    request_data = {
        "experiment_names": ["exp1", "exp2"]
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/experiments/compare", json=request_data)

    assert response.status_code == 200
    assert "comparison" in response.json()
    assert "exp1" in response.json()["comparison"]

@pytest.mark.asyncio
async def test_experiments_compare_empty(monkeypatch):
    fake_service = FakeMLService()
    monkeypatch.setattr("app.api.experiments.ml_service", fake_service)

    request_data = {
        "experiment_names": []
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/experiments/compare", json=request_data)

    assert response.status_code == 500 or response.status_code == 422
