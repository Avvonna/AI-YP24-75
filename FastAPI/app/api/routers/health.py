from app.core.ml_service import ml_pipeline

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["Monitoring"])

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.get("/info")
def info():
    return {"model": ml_pipeline.current_model}
