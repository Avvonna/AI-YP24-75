import streamlit as st
import pandas as pd
import requests
from json import dumps
from serializer import serialize_datetime
from matplotlib import pyplot as plt

from variables import BACKEND_URL

start_date = st.date_input('Enter start date')
end_date = st.date_input('Enter end date')
name = st.text_input('Enter ticker name')
if name and start_date and end_date:
    response = requests.post(BACKEND_URL+'/api/ticker-data', data=dumps({'ticker': name, 'start_date': start_date.isoformat(), 'end_date': end_date.isoformat()}, default=serialize_datetime))
    if response.status_code == 200:
        js = response.json()
        st.write('Ticker data:')
        st.write(pd.DataFrame(response.json()))
        fig, ax = plt.subplots()
        ax.plot(js['dates'], js['values'])
        plt.rcParams.update({'font.size': 5})
        st.pyplot(fig)