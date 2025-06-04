from typing import Annotated

from app.models.schemas import ModelSelectRequest
from app.services.ml_service import ml_service

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/model", tags=["Model"])

@router.post("/select")
async def select_model(request: Annotated[ModelSelectRequest, "Выбор модели"]):
    try:
        return ml_service.set_model(request.model_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
