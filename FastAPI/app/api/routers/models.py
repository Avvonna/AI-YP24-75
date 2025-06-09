from app.core.ml_service import ml_pipeline
from app.schemas import AvailableModelsResponse, ModelSchemaResponse, ModelSelectResponseSchema, ModelSelectSchema

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/models", tags=["Models"])

@router.get("/available", response_model=AvailableModelsResponse)
async def get_available_models():
    """Возвращает список всех доступных моделей"""
    try:
        return {"models": ml_pipeline.get_available_models()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.get("/parameters/{model_name}", response_model=ModelSchemaResponse)
async def get_model_parameters(model_name: str):
    """Возвращает JSON-схему параметров конфигурации модели"""
    try:
        if model_name not in ml_pipeline.get_available_models():
            raise HTTPException(status_code=404, detail=f"Модель '{model_name}' не найдена")

        config_schema = ml_pipeline.get_config_schema(model_name)

        return ModelSchemaResponse(
            model_name=model_name,
            config_schema=config_schema
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.post("/select", response_model=ModelSelectResponseSchema)
async def select_model(request: ModelSelectSchema):
    """Выбирает модель для использования в прогнозировании"""
    try:
        return ml_pipeline.set_model(request.model_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
