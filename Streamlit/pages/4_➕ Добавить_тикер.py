import os
import requests
from utils.logger import get_logger

import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
st.title("Добавить или обновить данные тикера")
logger = get_logger()
name = st.text_input('Введите имя будущего тикера')
file = st.file_uploader("Загрузить csv-файл", type="csv")
if st.button('Создать тикер'):
    response = requests.post(BACKEND_URL+'/api/tickers/', params={'ticker': name}, files={'file': file.getvalue()})
    if response.status_code == 200:
        st.write('Тикер создан!')
        logger.info(f"Created ticker {name}")
    else:
        st.warning("⚠️ Произошла ошибка. проверьте корректность введенных данных")
        logger.info(f"Failed to create ticker {name}")