import os
from json import dumps

import pandas as pd
import plotly.express as px
import requests
from serializer import serialize_datetime

import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

name = st.text_input("Enter ticker name")
start_date = st.date_input("Enter start date")
end_date = st.number_input("Enter forecast_period", value=10, min_value=1, max_value=100)
if st.button("Predict"):
    response = requests.post(
        url=BACKEND_URL+"/api/predict",
        data=dumps(
            {
                "ticker": name,
                "base_date": start_date.isoformat(),
                "forecast_period": end_date
                },
            default=serialize_datetime
            )
        )
    if response.status_code == 200:
        st.write("Prediction results")
        js = response.json()
        df = pd.DataFrame({"dates": js["forecast_dates"], "values": js["forecast_values"]})
        st.write(df)
        df = pd.DataFrame({"dates": js["forecast_dates"], "values": js["forecast_values"]})
        fig = px.line(df, x="dates", y="values")
        st.plotly_chart(fig, use_container_width=True)
