import streamlit as st
import pandas as pd
import requests
from json import dumps
import matplotlib.pyplot as plt
from serializer import serialize_datetime
import plotly.graph_objects as go
from variables import BACKEND_URL

name = st.text_input('Enter ticker name')
start_date = st.date_input('Enter start date')
end_date = st.number_input('Enter forecast_period', value=10, min_value=1, max_value=100)
if st.button('Predict'):
    response = requests.post(BACKEND_URL+'/api/predict', data=dumps({'ticker': name, 'base_date': start_date.isoformat(), 'forecast_period': end_date}, default=serialize_datetime))
    historical_data = requests.post(BACKEND_URL+'/api/ticker-data', data=dumps({'ticker': name, 'start_date': (pd.to_datetime(start_date) - pd.Timedelta(days=60)).isoformat(), 'end_date': start_date.isoformat()}, default=serialize_datetime))
    if response.status_code == 200:
        st.write('Prediction results')
        js = response.json()
        df = pd.DataFrame({'dates': js['forecast_dates'], 'values': js['forecast_values']})
        hist = pd.DataFrame({'dates': historical_data.json()['dates'], 'values': historical_data.json()['values']})
        st.write(df)
        df = pd.DataFrame({'dates': js['forecast_dates'], 'values': js['forecast_values']})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['dates'], y=df['values'], mode='lines', name='prediction'))
        fig.add_trace(go.Scatter(x=hist['dates'], y=hist['values'], mode='lines', line=dict(color='orange'), name='historical'))
        st.plotly_chart(fig, use_container_width=True)