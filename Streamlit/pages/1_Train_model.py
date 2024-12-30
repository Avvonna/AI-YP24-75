import streamlit as st
import pandas as pd
import requests
import logging
from json import dumps

from serializer import serialize_datetime
from variables import BACKEND_URL

logger = logging.getLogger(__name__)
name = st.text_input('Name of experiment')
seasonal = st.checkbox('seasonality')
if seasonal:
    seasonal_period = st.number_input('Seasonality period', value=7, min_value=1)
    config = {
        'seasonal': seasonal,
        'seasonal_period': seasonal_period,
    }
else:
    config = {
        'seasonal': seasonal
    }
max_p = st.number_input('max_p', value=5)
config.update({'max_p': max_p})
uploaded_file = st.file_uploader("Choose a file", type='csv')
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    df = pd.read_csv(uploaded_file, delimiter=';', dtype={'value': float})
    df['date'] = pd.to_datetime(df['date'])
    data = {
        'experiment_name': name,
        'dates': df['date'].tolist(),
        'values': df['value'].tolist()
    }
    train_response = requests.post(BACKEND_URL +'/api/train', data=dumps({'data': data, 'config': config}, default=serialize_datetime)).json()
    if train_response['status'] == 'успешно':
        st.write('Result of training (metrics)')
        st.write(train_response['metrics'])