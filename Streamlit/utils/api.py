import os

import requests
from utils.logger import get_logger

import streamlit as st

logger = get_logger()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


@st.cache_data(show_spinner="Загружаем тикеры...")
def load_tickers():
    try:
        response = requests.get(f"{BACKEND_URL}/api/tickers", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        logger.exception("Ошибка загрузки тикеров")
        st.error("Не удалось загрузить список тикеров")
        raise


@st.cache_data(show_spinner="Загружаем список моделей...")
def load_models():
    try:
        response = requests.get(f"{BACKEND_URL}/api/models/available", timeout=5)
        response.raise_for_status()
        return response.json().get("models", [])
    except Exception:
        logger.exception("Ошибка загрузки списка моделей")
        st.error("Не удалось загрузить список моделей")
        raise


@st.cache_data(show_spinner="Загружаем параметры модели...")
def load_model_schema(model_name: str):
    try:
        response = requests.get(f"{BACKEND_URL}/api/models/parameters/{model_name}", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        logger.exception(f"Ошибка загрузки схемы конфигурации модели '{model_name}'")
        st.error(f"Не удалось загрузить схему параметров модели: {model_name}")
        raise
