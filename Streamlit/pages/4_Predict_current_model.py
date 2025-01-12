import os

import pandas as pd
import requests
from StLogger import get_logger

import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")



logger = get_logger()
steps = st.number_input("Input number of steps", value=1)
max_p=st.number_input("Input max value of AR parameter", value=2, min_value=2)
max_d=st.number_input("Input max value of I parameter", value=2, min_value=2)
max_q=st.number_input("Input max value of MA parameter", value=2, min_value=2)
max_P=st.number_input("Input max value of seasonal AR parameter", value=3, min_value=3)
max_D=st.number_input("Input max value of seasonal I parameter", value=3, min_value=3)
max_Q=st.number_input("Input max value of seasonal MA parameter", value=3, min_value=3)
uploaded_file = st.file_uploader("Choose a file", type="csv")
st.write('Format of file is csv, delimited with ";" and two columns. Like this:')
st.write(pd.DataFrame({
    "date":pd.to_datetime(["2017-01-01", "2017-01-02"]),
    "value":[176.44, 199.99]
}))
send_btn = st.button("Train")
if send_btn:
    if max_P <= max_p or max_D <= max_d or max_Q <= max_q:
        st.write("All seasonal parameters must be greater than normal ones!")
    else:
        bytes_data = uploaded_file.getvalue()
        df = pd.read_csv(uploaded_file, delimiter=";", dtype={"value": float})
        data = {
            "steps": int(steps),
            "data": df["value"].tolist(),
            "config":{
                "max_p": max_p,
                "max_d": max_d,
                "max_q": max_q,
                "max_P": max_P,
                "max_D": max_D,
                "max_Q": max_Q,
            }
        }
        train_response = requests.post(
            url=BACKEND_URL +"/api/model/predict",
            json=data
            )
        if train_response.status_code == 200:
            st.write("Result of training forecast:")
            st.write(train_response.json()["forecast"][0])
            st.write("Lower end of confidence_intervals:")
            st.write(train_response.json()["confidence_intervals"])
            logger.info("Successfully predicted for current model")