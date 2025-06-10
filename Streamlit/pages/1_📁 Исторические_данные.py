from json import dumps

import pandas as pd
import plotly.express as px
import requests
from utils.api import BACKEND_URL, load_tickers
from utils.logger import get_logger

import streamlit as st

# Логгер
logger = get_logger()

# Загружаем тикеры
try:
    tickers = load_tickers()
except Exception:
    st.stop()

# UI
st.title("Исторические данные по тикеру")
name = st.selectbox("Выберите тикер", options=tickers)
start_date = st.date_input("Дата начала")
end_date = st.date_input("Дата окончания")

# Запрос данных
if st.button("Получить данные"):
    payload = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    }
    if start_date >= end_date:
        st.warning("❗ Начальная дата должна быть раньше конечной даты.")
        st.stop()

    try:
        response = requests.post(
            url=f"{BACKEND_URL}/api/tickers/{name}/history",
            data=dumps(payload),
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            js = response.json()
            df = pd.DataFrame({"dates": js["dates"], "values": js["values"]})
            df["dates"] = pd.to_datetime(df["dates"]).dt.date

            if df.empty:
                st.warning("⚠️ Нет данных за выбранный период. Проверьте корректность дат и наличие данных у тикера.")
                logger.warning(f"Пустой датафрейм для тикера {name} за период {start_date} — {end_date}")
                st.stop()
            else:
                actual_start = df["dates"].min()
                actual_end = df["dates"].max()
                if actual_start != start_date or actual_end != end_date:
                    st.info(
                        f"⚠️ Данные доступны только за период: с {actual_start} по {actual_end}. "
                        "Даты были автоматически скорректированы."
                    )

            st.success("Данные успешно получены")
            st.write("Таблица данных:")
            st.dataframe(df.round(1))

            st.write("Статистика (EDA):")
            st.write(df.describe().round(1))

            fig1 = px.box(df, y="values", title="Boxplot: цены акций")
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.histogram(df, x="values", labels={"values": "Цена"}, title="Histogram: цены акций")
            st.plotly_chart(fig2, use_container_width=True)

            fig3 = px.line(df.round(1), x="dates", y="values", labels={"dates": "Дата", "values": "Цена"},
                        title="Динамика цен по датам")
            st.plotly_chart(fig3, use_container_width=True)

            logger.info(f"Получены исторические данные по тикеру {name} за период с {start_date} по {end_date}")

        else:
            st.error(f"Ошибка API: {response.status_code} — {response.text}")
            logger.error(f"API error {response.status_code}: {response.text}")

    except Exception:
        st.error("Ошибка при запросе к API")
        logger.exception("Исключение при получении исторических данных")
