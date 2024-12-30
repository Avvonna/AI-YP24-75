import streamlit as st
import pandas as pd
import requests
from json import dumps
import matplotlib.pyplot as plt
from serializer import serialize_datetime
import plotly.express as px
from variables import BACKEND_URL

name = st.text_input('Enter ticker name')
start_date = st.date_input('Enter start date')
end_date = st.number_input('Enter forecast_period', value=10)
if name and start_date and end_date:
    response = requests.post(BACKEND_URL+'/api/predict', data=dumps({'ticker': name, 'base_date': start_date.isoformat(), 'forecast_period': end_date}, default=serialize_datetime))
    if response.status_code == 200:
        st.write('Результаты предсказания:')
        js = response.json()
        df = pd.DataFrame({'dates': js['forecast_dates'], 'values': js['forecast_values']})
        st.write(df)
        df = pd.DataFrame({'dates': js['forecast_dates'], 'values': js['forecast_values']})
        fig = px.line(df, x="dates", y="values")
        st.plotly_chart(fig, use_container_width=True)