import streamlit as st
from st_functions import st_button, load_css
from PIL import Image

load_css()


st.header('# Data Mining Application')

st.info('Developer Advocate, Content Creator and ex-Professor with an interest in Data Science and Bioinformatics')

icon_size = 20

st_button('streamlit', 'https://waterquality.streamlit.app/', 'Data Professor YouTube channel', icon_size)

