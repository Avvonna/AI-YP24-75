from app.models.schemas import ExperimentComparisonRequest
from app.services.ml_service import ml_service

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/experiments", tags=["Experiments"])

@router.post("/compare")
async def compare_experiments(request: ExperimentComparisonRequest):
    try:
        return ml_service.compare_experiments(request.experiment_names)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
