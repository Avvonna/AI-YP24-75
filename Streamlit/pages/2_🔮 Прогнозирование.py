import os
from datetime import date

import pandas as pd
import plotly.graph_objects as go
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
model = st.selectbox("Выберите модель", ["AutoARIMA", "CatBoost", "LSTM"])
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
elif model == "LSTM":
    st.markdown("**Параметры LSTM**")
    config = {
        "model_type": "lstm",
        "hidden_dim": st.number_input("Скрытых нейронов (hidden_dim)", min_value=1, value=64),
        "num_layers": st.number_input("Число слоёв (num_layers)", min_value=1, value=2),
        "dropout": st.number_input("Dropout", min_value=0.0, max_value=1.0, value=0.2),
        "lr": st.number_input("Скорость обучения (lr)", min_value=0.0001, value=0.001, format="%.4f"),
        "epochs": st.number_input("Эпохи обучения", min_value=1, value=5),
        "batch_size": st.number_input("Размер батча", min_value=1, value=32),
        "patience": st.number_input("Patience (ранняя остановка)", min_value=1, value=10),
        "window_size": st.number_input("Размер окна (window)", min_value=1, value=30),
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

        fig = go.Figure()

        # Историческая кривая
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["value"],
            mode="lines",
            name="история",
            line={"color": "blue", "dash": "solid"}
        ))

        # Прогнозная кривая
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["value"],
            mode="lines",
            name="прогноз",
            line={"color": "red", "dash": "dash"}
        ))

        fig.update_layout(
            title="История и прогноз",
            xaxis_title="Дата",
            yaxis_title="Значение",
            legend_title="Тип данных",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write("📎 Таблица прогноза:")
        st.dataframe(forecast_df)

    except Exception as e:
        st.error("Ошибка при построении прогноза")
        logger.exception(e)
