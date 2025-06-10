import pandas as pd
import requests
from utils.api import BACKEND_URL
from utils.logger import get_logger

import streamlit as st

# Настройки
logger = get_logger()

st.title("Сравнение экспериментов")

# Получение списка экспериментов
try:
    response = requests.get(f"{BACKEND_URL}/api/experiments/")
    response.raise_for_status()
    experiments = response.json()["experiments"]
except Exception as e:
    st.error(f"Не удалось загрузить список экспериментов: {e}")
    st.stop()

if not experiments:
    st.info("Эксперименты ещё не проводились.")
    st.stop()

records = []
for name in experiments:
    try:
        resp = requests.get(f"{BACKEND_URL}/api/experiments/{name}")
        resp.raise_for_status()
        data = resp.json()
        record = {
            "Имя": data["name"],
            "Модель": data["config"].get("model_type", "N/A"),
            "Тикер": data["training_data"].get("ticker", "N/A"),
            "Начало": min(data["training_data"].get("dates", []), default="—"),
            "Конец": max(data["training_data"].get("dates", []), default="—"),
            "MAE": data["metrics"].get("mae"),
            "MSE": data["metrics"].get("mse"),
            "AIC": data["metrics"].get("aic"),
            "BIC": data["metrics"].get("bic"),
            "Эпохи": data["config"].get("epochs"),
            "Batch Size": data["config"].get("batch_size"),
        }
        records.append(record)
    except Exception as e:
        st.warning(f"Ошибка загрузки эксперимента {name}: {e}")

if not records:
    st.warning("Нет данных для отображения.")
    st.stop()

df = pd.DataFrame(records)

st.subheader("📋 Таблица экспериментов")
st.dataframe(df.round(2))

# Визуализация по выбранной метрике
st.subheader("📊 Визуализация метрик")

available_metrics = ["MAE", "MSE", "AIC", "BIC"]
selected_metric = st.selectbox("Выберите метрику для отображения", available_metrics)

metric_series = df.set_index("Имя")[selected_metric].dropna()

if metric_series.empty:
    st.info(f"Нет доступных значений для метрики '{selected_metric}'")
else:
    st.bar_chart(metric_series)

