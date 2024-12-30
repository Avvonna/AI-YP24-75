import streamlit as st
import pandas as pd
import requests
from json import dumps
from serializer import serialize_datetime
from matplotlib import pyplot as plt
import plotly.express as px
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

start_date = st.date_input('Enter start date')
end_date = st.date_input('Enter end date')
name = st.text_input('Enter ticker name')
if st.button('Get data'):
    response = requests.post(BACKEND_URL+'/api/ticker-data', data=dumps({'ticker': name, 'start_date': start_date.isoformat(), 'end_date': end_date.isoformat()}, default=serialize_datetime))
    if response.status_code == 200:
        js = response.json()
        st.write('Ticker data:')
        df = pd.DataFrame({'dates': js['dates'], 'values': js['values']})
        st.write(df)
        fig = px.line(df, x="dates", y="values")
        st.plotly_chart(fig, use_container_width=True)