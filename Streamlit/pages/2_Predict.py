import os
import pandas as pd
import plotly.express as px
import requests
from json import dumps
from serializer import serialize_datetime
from StLogger import get_logger

import streamlit as st
import plotly.graph_objects as go

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

logger = get_logger()

tickers = requests.get(BACKEND_URL+"/api/tickers").content.decode()
name = st.selectbox("Select ticker name", options=eval(tickers))
start_date = st.date_input("Enter start date")
end_date = st.number_input("Enter forecast_period", value=10, min_value=1, max_value=100)
if st.button('Predict'):
    response = requests.post(BACKEND_URL+'/api/predict', data=dumps({'ticker': name, 'base_date': start_date.isoformat(), 'forecast_period': end_date}, default=serialize_datetime))
    historical_data = requests.post(BACKEND_URL+'/api/ticker-data', data=dumps({'ticker': name, 'start_date': (pd.to_datetime(start_date) - pd.Timedelta(days=60)).isoformat(), 'end_date': start_date.isoformat()}, default=serialize_datetime))
    if response.status_code == 200:
        st.write("Prediction results")
        js = response.json()
        df = pd.DataFrame({"dates": js["forecast_dates"], "values": js["forecast_values"]})
        df = pd.DataFrame({'dates': js['forecast_dates'], 'values': js['forecast_values']})
        hist = pd.DataFrame({'dates': historical_data.json()['dates'], 'values': historical_data.json()['values']})
        st.write(df)
        df = pd.DataFrame({"dates": js["forecast_dates"], "values": js["forecast_values"]})
        fig = px.line(df, x="dates", y="values")
        df = pd.DataFrame({'dates': js['forecast_dates'], 'values': js['forecast_values']})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['dates'], y=df['values'], mode='lines', name='prediction'))
        fig.add_trace(go.Scatter(x=hist['dates'], y=hist['values'], mode='lines', line=dict(color='orange'), name='historical'))
        st.plotly_chart(fig, use_container_width=True)
        logger.info(f'Predicted sucessfully for ticker {name}, for {end_date} days from {start_date}')
