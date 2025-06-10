from app.core.ml_service import ml_pipeline
from app.schemas import ComparisonResult, ExperimentComparisonSchema, ExperimentRecord

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/experiments", tags=["Experiments"])


@router.get("/", response_model=dict[str, list[str]])
async def list_experiments():
    """Возвращает список всех сохранённых экспериментов"""
    try:
        return {"experiments": ml_pipeline.get_experiments_list()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.get("/{name}", response_model=ExperimentRecord)
async def get_experiment(name: str):
    try:
        result = ml_pipeline.get_experiment(name)
        if not result:
            raise HTTPException(status_code=404, detail=f"Эксперимент '{name}' не найден")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.post("/compare", response_model=ComparisonResult)
async def compare_experiments(request: ExperimentComparisonSchema):
    try:
        return ml_pipeline.compare_experiments(request.experiment_names)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
