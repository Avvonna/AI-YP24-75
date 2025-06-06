from typing import Any

from app.core.ml_service import ml_pipeline
from app.schemas import ExperimentComparisonRequest

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/experiments", tags=["Experiments"])

@router.post("/compare", response_model=dict[str, Any])
async def compare_experiments(request: ExperimentComparisonRequest):
    try:
        return ml_pipeline.compare_experiments(request.experiment_names)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
