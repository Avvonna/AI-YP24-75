import streamlit as st

st.write("This is boring main page with index")

st.markdown(
    """

    ### Said index:
    - [Train model](/Train_model)
    - [Make prediction](/Predict)
    - [Get data from particular ticket](/Get_ticket_data)
    - [Compare experiments](/Compare_experiments)
    - [Save or load model](/Load_model)
"""
)
