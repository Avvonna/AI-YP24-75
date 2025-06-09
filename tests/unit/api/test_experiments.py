import app.api.routers.experiments as experiments_module
import pytest
from app.api.routers.experiments import router as experiments_router
from app.configs import AutoARIMAConfig
from app.schemas import ComparisonResult, ExperimentMetrics, ExperimentRecord, TickerHistory
from httpx import ASGITransport, AsyncClient

from fastapi import FastAPI

app = FastAPI()
app.include_router(experiments_router)


class FakeMLService:
    def compare_experiments(self, names: list[str]) -> ComparisonResult:
        if not names:
            raise ValueError("Список имён экспериментов пуст.")

        experiments = [
            ExperimentRecord(
                name=n,
                model=None,
                config=AutoARIMAConfig().model_dump(),
                metrics=ExperimentMetrics(mae=1.0, mse=2.0),
                training_data=TickerHistory(ticker=n, dates=["2023-01-01"], values=[100.0])
            )
            for n in names
        ]

        return ComparisonResult(experiments=experiments, missing_experiments=[])

@pytest.mark.asyncio
async def test_experiments_compare_success(monkeypatch):
    monkeypatch.setattr(experiments_module, "ml_pipeline", FakeMLService())

    request_data = {"experiment_names": ["exp1", "exp2"]}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/experiments/compare", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data
    assert len(data["experiments"]) == 2

@pytest.mark.asyncio
async def test_experiments_compare_empty(monkeypatch):
    fake_service = FakeMLService()
    monkeypatch.setattr("app.api.routers.experiments.ml_pipeline", fake_service)

    request_data = {"experiment_names": []}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/experiments/compare", json=request_data)

    assert response.status_code in (422, 400)
