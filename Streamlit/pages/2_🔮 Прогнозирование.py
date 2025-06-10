from datetime import date

import pandas as pd
import plotly.graph_objects as go
import requests
from utils.api import BACKEND_URL, load_model_schema, load_models, load_tickers
from utils.logger import get_logger
from utils.render_config import render_config_from_schema

import streamlit as st

# Настройки
logger = get_logger()

st.title("Прогнозирование по тикерам")

# Инициализация session state
if "is_forecasting" not in st.session_state:
    st.session_state.is_forecasting = False
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None

# 1. Загрузка тикеров и моделей
try:
    tickers = load_tickers()
    models = load_models()
except Exception:
    st.stop()

# 2. Выбор тикера и интервала
ticker = st.selectbox("Выберите тикер", tickers)
col1, col2 = st.columns(2)
start_date = col1.date_input("Начальная дата", value=date(2022, 1, 1))
end_date = col2.date_input("Конечная дата", value=date.today())

if end_date <= start_date:
    st.warning("❗ Конечная дата должна быть позже начальной.")
    st.stop()

# 3. Параметры модели
model = st.selectbox("Выберите модель", models)
forecast_period = st.number_input("Горизонт прогноза", min_value=1, value=10)

st.markdown(f"**Параметры {model}**")

# 3.1 Получение схемы конфигурации и генерация формы
try:
    schema = load_model_schema(model)["config_schema"]
except Exception:
    st.stop()
config = render_config_from_schema(schema)

# 4. Кнопка запуска с блокировкой
if st.session_state.is_forecasting:
    st.info("🔄 Выполняется прогноз... Пожалуйста, подождите.")
    st.button("🔮 Построить прогноз", disabled=True)
else:
    forecast_button = st.button("🔮 Построить прогноз")

    if forecast_button:
        st.session_state.is_forecasting = True
        st.rerun()

# 5. Выполнение прогноза
if st.session_state.is_forecasting:
    try:
        with st.spinner("Загрузка данных и построение прогноза..."):
            # Получение истории
            history_resp = requests.post(
                f"{BACKEND_URL}/api/tickers/{ticker}/history",
                json={"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
            )
            history = history_resp.json()
            df = pd.DataFrame({"date": history["dates"], "value": history["values"]})

            if df.empty:
                st.warning("Нет данных за выбранный период для выбранного тикера.")
                st.session_state.is_forecasting = False
                st.rerun()

            df["date"] = pd.to_datetime(df["date"]).dt.floor("D")

            actual_start = df["date"].min().date()
            actual_end = df["date"].max().date()

            if actual_start != start_date or actual_end != end_date:
                st.info(
                    f"⚠️ Данные доступны только за период: с {actual_start} по {actual_end}. "
                    "Даты были автоматически скорректированы."
                )

            # Прогноз
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
                st.session_state.is_forecasting = False
                st.rerun()

            # Сохраняем результат в session state
            forecast_df = pd.DataFrame({
                "date": pd.to_datetime(result["forecast_dates"]).map(lambda d: d.date()),
                "value": result["forecast_values"]
            })

            st.session_state.forecast_result = {
                "df": df,
                "forecast_df": forecast_df,
                "actual_start": actual_start,
                "actual_end": actual_end
            }

            # Завершаем операцию
            st.session_state.is_forecasting = False
            st.rerun()

    except Exception as e:
        st.error("Ошибка при построении прогноза")
        logger.exception(e)
        st.session_state.is_forecasting = False
        st.rerun()

# 6. Отображение результатов
if st.session_state.forecast_result:
    result_data = st.session_state.forecast_result

    st.success("Прогноз выполнен!")
    st.subheader("📊 История и прогноз")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result_data["df"]["date"],
        y=result_data["df"]["value"],
        mode="lines",
        name="история"
    ))
    fig.add_trace(go.Scatter(
        x=result_data["forecast_df"]["date"],
        y=result_data["forecast_df"]["value"],
        mode="lines",
        name="прогноз",
        line={"dash": "dash"}
    ))
    fig.update_layout(
        title="История и прогноз",
        xaxis_title="Дата",
        yaxis_title="Значение",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.write("📎 Таблица прогноза:")
    st.dataframe(result_data["forecast_df"].round(1))

    # Кнопка для очистки результатов
    if st.button("🗑️ Очистить результаты"):
        st.session_state.forecast_result = None
        st.rerun()
