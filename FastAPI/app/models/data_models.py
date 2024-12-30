from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field, validator

class TimeSeriesData(BaseModel):
   """Модель временного ряда"""
   dates: List[datetime] = Field(..., description="Список дат временного ряда")
   values: List[float] = Field(..., description="Список значений временного ряда") 
   experiment_name: str = Field(..., description="Уникальное имя эксперимента")

   @validator('values')
   def validate_values(cls, v: List[float]) -> List[float]:
       if not v:
           raise ValueError("Список значений не может быть пустым")
       if not all(isinstance(x, (int, float)) or np.isnan(x) for x in v):
           raise ValueError("Все значения должны быть числовыми") 
       return v

class ModelConfig(BaseModel):
   """Параметры конфигурации модели"""
   seasonal: bool = Field(default=True, description="Учитывать ли сезонность")
   seasonal_period: Optional[int] = Field(default=7, ge=1, description="Период сезонности")
   max_p: int = Field(default=5, description="Максимальный порядок AR члена")

   @validator('seasonal_period')
   def validate_seasonal_period(cls, v: Optional[int], values: Dict[str, Any]) -> Optional[int]:
       if values.get('seasonal') and (v is None or v < 1):
           raise ValueError("Период сезонности должен быть положительным числом при seasonal=True")
       return v

class ModelSelectRequest(BaseModel):
   """Запрос на выбор модели"""
   model_name: str = Field(..., description="Название модели")

class PredictRequest(BaseModel):
   """Запрос на прогнозирование"""
   ticker: str = Field(..., description="Название тикера")
   base_date: str = Field(..., description="Дата для прогноза")
   forecast_period: int = Field(default=10, ge=1, description="Горизонт прогнозирования")

   @validator('forecast_period')
   def validate_forecast_period(cls, v: int) -> int:
       if v < 1:
           raise ValueError("Период прогнозирования должен быть положительным") 
       return v

class CurrentModelPredictRequest(BaseModel):
   """Запрос прогноза текущей модели"""
   data: List[float] = Field(..., min_items=60, description="Исторические данные")
   steps: int = Field(default=10, ge=1, description="Количество шагов прогноза")

   @validator('data')
   def validate_data(cls, v: List[float]) -> List[float]:
       if not all(isinstance(x, (int, float)) for x in v):
           raise ValueError("Все значения должны быть числовыми")
       return v

class HistoricalDataRequest(BaseModel):
   """Запрос исторических данных"""
   start_date: str = Field(..., description="Начальная дата")
   end_date: str = Field(..., description="Конечная дата")

   @validator('end_date')
   def validate_dates(cls, v: str, values: Dict[str, Any]) -> str:
       start = datetime.fromisoformat(values['start_date'])
       end = datetime.fromisoformat(v)
       if end <= start:
           raise ValueError("Конечная дата должна быть позже начальной")
       return v

class HistoricalDataResponse(BaseModel):
   """Ответ с историческими данными"""
   dates: List[str] = Field(..., description="Список дат")
   values: List[float] = Field(..., description="Список значений")

   @validator('dates')
   def validate_dates(cls, v: List[str]) -> List[str]:
       for date in v:
           try:
               datetime.fromisoformat(date)
           except ValueError:
               raise ValueError(f"Некорректный формат даты: {date}")
       return v

class ExperimentMetrics(BaseModel):
   """Метрики эксперимента"""
   aic: float = Field(..., description="Информационный критерий Акаике")
   bic: float = Field(..., description="Байесовский информационный критерий")
   mse: float = Field(..., description="Средний квадрат ошибки")
   mae: float = Field(..., description="Средняя абсолютная ошибка")

class ExperimentTrainingHistory(BaseModel):
   """История обучения эксперимента"""
   dates: List[str] = Field(..., description="Даты обучения")
   values: List[float] = Field(..., description="Значения обучения")

class ExperimentData(BaseModel):
   """Данные эксперимента"""
   metrics: ExperimentMetrics
   config: ModelConfig
   training_history: ExperimentTrainingHistory

class ExperimentComparisonRequest(BaseModel):
   """Запрос на сравнение экспериментов"""
   experiment_names: List[str] = Field(..., min_items=1, description="Список имён экспериментов для сравнения")

   @validator('experiment_names')
   def validate_experiment_names(cls, v: List[str]) -> List[str]:
       if len(v) < 1:
           raise ValueError("Требуется хотя бы одно имя эксперимента")
       if len(v) != len(set(v)):
           raise ValueError("Дублирование имен экспериментов не допускается")
       return v

class ExperimentComparisonResponse(BaseModel):
   """Ответ сравнения экспериментов"""
   status: Optional[str] = Field(None, description="Статус выполнения")
   message: Optional[str] = Field(None, description="Сообщение о результате")
   experiments: Dict[str, ExperimentData] = Field(default_factory=dict, description="Данные экспериментов")
   missing_experiments: Optional[Dict[str, Any]] = Field(None, description="Информация о ненайденных экспериментах")