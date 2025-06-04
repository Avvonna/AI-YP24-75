import pytest
from app.configs import AutoARIMAConfig
from app.main import app
from httpx import ASGITransport, AsyncClient


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
        response = await ac.post("/api/predictions/", json=request_data)

    assert response.status_code == 200
    json_data = response.json()
    assert "forecast_values" in json_data
    assert len(json_data["forecast_values"]) == 2
