import os
from datetime import date

import pandas as pd
import plotly.express as px
import requests
from utils.logger import get_logger

import streamlit as st

# Настройки
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
logger = get_logger()

st.title("📈 Прогнозирование по тикерам")

# 1. Выбор тикера и интервала
@st.cache_data(show_spinner="Загружаем тикеры...")
def load_tickers():
    return requests.get(f"{BACKEND_URL}/api/tickers").json()

tickers = load_tickers()
ticker = st.selectbox("Выберите тикер", tickers)
col1, col2 = st.columns(2)
start_date = col1.date_input("Начальная дата", value=date(2022, 1, 1))
end_date = col2.date_input("Конечная дата", value=date.today())

if end_date <= start_date:
    st.warning("❗ Конечная дата должна быть позже начальной.")
    st.stop()

# 2. Параметры модели
model = st.selectbox("Выберите модель", ["AutoARIMA", "CatBoost"])
forecast_period = st.number_input("Горизонт прогноза", min_value=1, value=10)

config = {}
if model == "AutoARIMA":
    st.markdown("**Параметры AutoARIMA**")
    config = {
        "model_type": "auto_arima",
        "max_p": st.number_input("max_p", value=2),
        "max_d": st.number_input("max_d", value=1),
        "max_q": st.number_input("max_q", value=2),
        "max_P": st.number_input("max_P", value=2),
        "max_D": st.number_input("max_D", value=1),
        "max_Q": st.number_input("max_Q", value=2),
        "seasonal": st.checkbox("Сезонность", value=True),
        "seasonal_period": st.number_input("Сезонный период", value=7),
    }
elif model == "CatBoost":
    st.markdown("**Параметры CatBoost**")
    config = {
        "model_type": "catboost",
        "iterations": st.number_input("Итерации", value=300),
        "learning_rate": st.number_input("Скорость обучения", value=0.03),
        "depth": st.number_input("Глубина", value=6),
    }

# 3. Кнопка запуска
if st.button("🔮 Построить прогноз"):
    try:
        # 1. Получение истории
        history_resp = requests.post(
            f"{BACKEND_URL}/api/tickers/{ticker}/history",
            json={"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
        )
        history = history_resp.json()
        df = pd.DataFrame({"date": history["dates"], "value": history["values"]})
        if df.empty:
            st.warning("Нет данных за выбранный период для выбранного тикера.")
            st.stop()
        df["date"] = pd.to_datetime(df["date"]).dt.floor("D")

        # 1.1 Проверка реального интервала данных
        actual_start = df["date"].min().date()
        actual_end = df["date"].max().date()

        if actual_start != start_date or actual_end != end_date:
            st.info(
                f"⚠️ Данные доступны только за период: с {actual_start} по {actual_end}. "
                "Даты были автоматически скорректированы."
            )

        # 2. Отправка на обучение и прогноз
        request_payload = {
            "ticker": ticker,
            "base_date": actual_end.isoformat(),
            "forecast_period": forecast_period,
            "config": config,
        }

        predict_resp = requests.post(f"{BACKEND_URL}/api/predictions", json=request_payload)
        result = predict_resp.json()

        if predict_resp.status_code != 200 or "forecast_values" not in result:
            st.error(f"Ошибка от backend: {result}")
            logger.error(f"Ошибка от backend: {result}")
            st.stop()

        # 3. Визуализация
        forecast_df = pd.DataFrame({
            "date": pd.to_datetime(result["forecast_dates"]).floor("D"),
            "value": result["forecast_values"]
        })

        st.success("Прогноз выполнен!")
        st.subheader("📊 История и прогноз")

        combined_df = pd.concat([
            df[["date", "value"]].rename(columns={"value": "value"}),
            forecast_df.rename(columns={"forecast": "value"})
        ])
        combined_df["type"] = ["история"] * len(df) + ["прогноз"] * len(forecast_df)

        fig = px.line(combined_df, x="date", y="value", color="type", title="История и прогноз")
        st.plotly_chart(fig, use_container_width=True)

        st.write("📎 Таблица прогноза:")
        st.dataframe(forecast_df)

    except Exception as e:
        st.error("Ошибка при построении прогноза")
        logger.exception(e)
