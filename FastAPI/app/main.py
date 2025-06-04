import warnings

from app.api import experiments, health, models, predict, tickers
from app.utils.logging_config import setup_logging

from fastapi import FastAPI

warnings.filterwarnings("ignore", message="'force_all_finite' was renamed")

logger = setup_logging()

app = FastAPI(
    title="Model Forecasting API",
    description="API для выбора моделей, предсказаний и анализа экспериментов",
    version="1.0.0"
)

app.include_router(tickers.router)
app.include_router(models.router)
app.include_router(predict.router)
app.include_router(experiments.router)
app.include_router(health.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
