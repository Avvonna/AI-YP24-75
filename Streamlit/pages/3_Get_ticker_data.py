import os
from json import dumps
import pandas as pd
import plotly.express as px
import requests
from serializer import serialize_datetime

import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

tickers = requests.get(BACKEND_URL+"/api/tickers").content.decode()

start_date = st.date_input("Enter start date")
end_date = st.date_input("Enter end date")
name = st.selectbox("Select ticker name", options=eval(tickers))
if st.button("Get data"):
    response = requests.get(
        url=BACKEND_URL+f"/api/tickers/{name}/history",
        data=dumps(
            {
                "start_date": str(start_date.isoformat()),
                "end_date": str( end_date.isoformat())
                },
            default=serialize_datetime
            )
        )

    if response.status_code == 200:
        js = response.json()
        st.write("Ticker data:")
        df = pd.DataFrame({"dates": js["dates"], "values": js["values"]})
        st.write(df)
        st.write("For eda:")
        st.write(df.describe())
        fig_1 = px.box(df, y='values')
        st.plotly_chart(fig_1, use_container_width=True)
        fig_2 = px.histogram(df, x="values")
        st.plotly_chart(fig_2, use_container_width=True)
        fig = px.line(df, x="dates", y="values")
        st.plotly_chart(fig, use_container_width=True)
