from app.services.ml_service import ml_service

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["Monitoring"])

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.get("/info")
def info():
    return {"model": ml_service.current_model}
