import os

import requests

import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

names = st.text_area("Input names of experiments (divide by commas)")
if names:
    st.write({i.strip() for i in names.split(",")})
    response = requests.post(BACKEND_URL+"/api/compare", json={"experiment_names": names})
    st.write(response.json())
