import os
from json import dumps
import pandas as pd
import plotly.express as px
import requests
from serializer import serialize_datetime

import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

name = st.text_input('Input ticker name')

if st.button('Create ticker'):
    response = requests.post(BACKEND_URL+'/api/tickers/add', params={'ticker': name})
    if response.status_code == 200:
        st.write('Created!')