import logging
import os
from json import dumps

import pandas as pd
import plotly.express as px
import requests
from serializer import serialize_datetime
from StLogger import get_logger

import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

name = st.text_input('Input ticker name')
logger = get_logger()
if st.button('Create ticker'):
    response = requests.post(BACKEND_URL+'/api/tickers/add', params={'ticker': name})
    if response.status_code == 200:
        st.write('Created!')
        logger.info(f"Created ticker {name}")
    else:
        logger.info(f"Failed to create ticker {name}")