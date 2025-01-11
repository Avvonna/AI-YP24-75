import os
from json import dumps

import pandas as pd
import plotly.express as px
import requests

from StLogger import get_logger

import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

logger = get_logger()

tickers = requests.get(BACKEND_URL+"/api/tickers").content.decode()

start_date = st.date_input("Enter start date")
end_date = st.date_input("Enter end date")
name = st.selectbox("Select ticker name", options=eval(tickers))
if st.button("Get data"):
    response = requests.post(
        url=BACKEND_URL+f"/api/tickers/{name}/history",
        data=dumps(
            {
                "ticker": name,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
                },
            )
        )

    if response.status_code == 200:
        js = response.json()
        st.write("Ticker data:")
        df = pd.DataFrame({"dates": js["dates"], "values": js["values"]})
        st.write(df)
        st.write("For eda:")
        st.write(df.describe())
        fig_1 = px.box(df, y="values", title="Boxplot for costs of ticker shares")
        st.plotly_chart(fig_1, use_container_width=True)
        fig_2 = px.histogram(df, x="values", labels={"values": "Cost"}, title='Histogram for costs of ticker shares')
        st.plotly_chart(fig_2, use_container_width=True)
        fig = px.line(df, x="dates", y="values", labels={"dates": "Date", "values": "Cost"},
            title='Costs of ticker shares by date')
        st.plotly_chart(fig, use_container_width=True)
        logger.info(f"Get ticker data for {name}")