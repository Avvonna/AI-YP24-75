import os
import sys

import pytest
from httpx import ASGITransport, AsyncClient

from fastapi import FastAPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../FastAPI")))

from app.api.health import router as health_router
from app.services.ml_service import ml_service

app = FastAPI()
app.include_router(health_router)


@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_info_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/info")

    assert response.status_code == 200
    assert "model" in response.json()
    assert response.json()["model"] == ml_service.current_model
