import streamlit as st
import numpy as np
import pandas as pd

st.title('Data Mining')
st.write("""
## Sistem Pendukung Keputusan Kelayakan Air Minum 
 """)

nama_dataset = st.sidebar.selectbox(
    'Choose Dataset',
    ('Water Potability','Air Quality')
 )

st.write(f"### Dataset {nama_dataset}")

algoritma = st.sidebar.selectbox(
    'Choose Algorithm',
    ('K-NN','Naive Bayes')
)


        