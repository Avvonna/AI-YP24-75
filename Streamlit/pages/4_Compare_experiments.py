import streamlit as st
import pandas as pd
import requests
from json import dumps
from serializer import serialize_datetime
from matplotlib import pyplot as plt

from variables import BACKEND_URL

names = st.text_area("Input names of experiments (divide by commas)")
if names:
    st.write(set([i.strip() for i in names.split(',')]))
    response = requests.post(BACKEND_URL+"/api/compare", json={'experiment_names': names})
    st.write(response.json())