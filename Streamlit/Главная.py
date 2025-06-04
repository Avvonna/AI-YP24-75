import streamlit as st

st.set_page_config(
    page_title="Аналитика акций",
    page_icon="📈",
    layout="centered",
)

st.title("Добро пожаловать в аналитическую платформу по прогнозированию цен на акции")

st.markdown(
    """
    Это приложение позволяет:

    - ...
    - ...

    ---
    ### 📂 Навигация:
    """
)

# Навигационные кнопки (указываем путь к файлам в папке /pages)
col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/1_📁 Исторические_данные.py", label="Исторические данные", icon="📈")

with col2:
    st.page_link("pages/2_🔮 Прогнозирование.py", label="Построить прогноз", icon="🔮")
