import logging
import os
from json import dumps

import pandas as pd
import requests
from serializer import serialize_datetime

import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")



logger = logging.getLogger(__name__)
steps = st.number_input("Input number of steps", value=1)
uploaded_file = st.file_uploader("Choose a file", type="csv")
st.write('Format of file is csv, delimited with ";" and two columns. Like this:')
st.write(pd.DataFrame({
    "date":pd.to_datetime(["2017-01-01", "2017-01-02"]),
    "value":[176.44, 199.99]
}))
send_btn = st.button("Train")
if send_btn:
    bytes_data = uploaded_file.getvalue()
    df = pd.read_csv(uploaded_file, delimiter=";", dtype={"value": float})
    data = {
        "steps": int(steps),
        "data": df["value"].tolist()
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
