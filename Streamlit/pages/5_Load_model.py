import streamlit as st
import pandas as pd
import requests
from json import dumps

from tornado.options import options

from serializer import serialize_datetime
from matplotlib import pyplot as plt

from variables import BACKEND_URL

action = st.radio('Load or save model?', options=['load', 'save'])
path = st.text_input('Input path to save or load model')
name = st.text_input('Input name of experiment')
model_format = st.radio('Choose format of file', options=['pickle', 'joblib'])
if action == 'save':
    if st.button('Save'):
        response = requests.post(BACKEND_URL+'/api/model/save', params={'path': path, 'experiment_name': name, 'format_type': model_format})
        if response.status_code == 200:
            st.write('Saved successfully')
else:
    if st.button('Load'):
        response = requests.post(BACKEND_URL+'/api/model/load', params={'path': path, 'experiment_name': name, 'format_type': model_format})
        if response.status_code == 200:
            st.write('Loaded successfully')