import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib
from sklearn.preprocessing import StandardScaler
from PIL import Image


st.set_page_config(page_title="Diah Kamalia")

st.title("Aplikasi Data Mining - Kualitas Air")
st.write("Diah Kamalia - 200411100061")

desc, dataset, preprocessing, modelling, implementation = st.tabs(["Deskripsi Data", "Dataset", "Preprocessing", "Modelling", "Implementasi"])

with desc:

    st.write("## About Dataset")
    
    st.write("## Context")
    st.write("""Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection. This is important as a health and development issue at a national, regional and local level. In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions.
            """)

    st.write("## Content")
    st.write("""
    1. pH value:
        > PH is an important parameter in evaluating the acid–base balance of water. It is also the indicator of acidic or alkaline condition of water status. WHO has recommended maximum permissible limit of pH from 6.5 to 8.5. The current investigation ranges were 6.52–6.83 which are in the range of WHO standards.

    2.  Hardness:
        > Hardness is mainly caused by calcium and magnesium salts. These salts are dissolved from geologic deposits through which water travels. The length of time water is in contact with hardness producing material helps determine how much hardness there is in raw water. Hardness was originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.

    3. Solids (Total dissolved solids - TDS):
        > Water has the ability to dissolve a wide range of inorganic and some organic minerals or salts such as potassium, calcium, sodium, bicarbonates, chlorides, magnesium, sulfates etc. These minerals produced un-wanted taste and diluted color in appearance of water. This is the important parameter for the use of water. The water with high TDS value indicates that water is highly mineralized. Desirable limit for TDS is 500 mg/l and maximum limit is 1000 mg/l which prescribed for drinking purpose.

    4. Chloramines:
        > Chlorine and chloramine are the major disinfectants used in public water systems. Chloramines are most commonly formed when ammonia is added to chlorine to treat drinking water. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.

    5. Sulfate:
        > Sulfates are naturally occurring substances that are found in minerals, soil, and rocks. They are present in ambient air, groundwater, plants, and food. The principal commercial use of sulfate is in the chemical industry. Sulfate concentration in seawater is about 2,700 milligrams per liter (mg/L). It ranges from 3 to 30 mg/L in most freshwater supplies, although much higher concentrations (1000 mg/L) are found in some geographic locations.

    6. Conductivity:
        > Pure water is not a good conductor of electric current rather’s a good insulator. Increase in ions concentration enhances the electrical conductivity of water. Generally, the amount of dissolved solids in water determines the electrical conductivity. Electrical conductivity (EC) actually measures the ionic process of a solution that enables it to transmit current. According to WHO standards, EC value should not exceeded 400 μS/cm.

    7. Organic Carbon:
        > Total Organic Carbon (TOC) in source waters comes from decaying natural organic matter (NOM) as well as synthetic sources. TOC is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is use for treatment.

    8. Trihalomethanes:
        > THMs are chemicals which may be found in water treated with chlorine. The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm is considered safe in drinking water.

    9. Turbidity:
        > The turbidity of water depends on the quantity of solid matter present in the suspended state. It is a measure of light emitting properties of water and the test is used to indicate the quality of waste discharge with respect to colloidal matter. The mean turbidity value obtained for Wondo Genet Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU.

    10. Potability:
        > Indicates if water is safe for human consumption where 1 means Potable and 0 means Not potable.
            """)

    st.write("## Column Description")
    st.write("""
    1. ph: pH of 1. water (0 to 14).
    2. Hardness: Capacity of water to precipitate soap in mg/L.
    3. Solids: Total dissolved solids in ppm.
    4. Chloramines: Amount of Chloramines in ppm.
    5. Sulfate: Amount of Sulfates dissolved in mg/L.
    6. Conductivity: Electrical conductivity of water in μS/cm.
    7. Organic_carbon: Amount of organic carbon in ppm.
    8. Trihalomethanes: Amount of Trihalomethanes in μg/L.
    9. Turbidity: Measure of light emiting property of water in NTU.
    10. Potability: Indicates if water is safe for human consumption. Potable -1 and Not potable -0
    """)
    
    st.write("## Dataset Source")
    st.write("Dataset Water Potability Dari Kaggle")
    kaggle = "https://www.kaggle.com/datasets/adityakadiwal/water-potability"
    st.markdown(f'[Dataset Water Potability - Kaggle ]({kaggle})')
    st.write("Dataset Water Quality Dari Github")
    github = "https://raw.githubusercontent.com/diahkamalia/waterquality/main/water_potability.csv"
    st.markdown(f'[Dataset Water Potability - Github ]({github})')
    st.write("## Repository Github")
    st.write(" Click the link below to access the source code")
    repo = "https://github.com/diahkamalia/waterquality"
    st.markdown(f'[ Link Repository Github ]({repo})')


with dataset:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        df = df.dropna()
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)
        
        
with preprocessing : 
    st.write("""# Preprocessing""")
    st.write("Data yang belum dinormalisasi")
    st.dataframe(df)

    label = df["Potability"]

    st.write("## Drop Column")
    st.write("Dropping 'Potability' Table")
    X = df.drop(columns=["Potability"])

    st.dataframe(X)

    st.write(" ## Normalisasi ")
    st.write(""" > ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity """)
    label_lama = ['ph', 'Hardness', 'Solids', 'Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
    label_baru = ['norm_ph', 'norm_Hardness', 'norm_Solids', 'norm_Chloramines','norm_Sulfate','norm_Conductivity','norm_Organic_carbon','norm_Trihalomethanes','norm_Turbidity']
    normalisasi_kolom = df[label_lama]

    st.dataframe(normalisasi_kolom)

    scaler = MinMaxScaler()
    scaler.fit(normalisasi_kolom)
    kolom_ternormalisasi = scaler.transform(normalisasi_kolom)
    df_kolom_ternormalisasi = pd.DataFrame(kolom_ternormalisasi, columns = label_baru)

    st.write("### Data After Normalized")
    st.dataframe(df_kolom_ternormalisasi)

    X = X.drop(columns = label_lama)

    X = X.join(df_kolom_ternormalisasi)

    X = X.join(label)

    st.write("dataframe X baru")
    st.dataframe(X)

    st.write("## Hitung Data")
    st.write("- Drop Potability column from dataframe")
    st.write("- Split Data Training and Data Test")
    st.write(""" ### Spliting Data
            > Data Training - X_train 
            > Data Test - X_test 
            > Data Training (Class) - y_train
            > Data Test (Class) - y_test
            """)


    # memisahkan data Potability
    X = X.iloc[:]
    y = df.loc[:, "Potability"]
    y = df["Potability"].values

    # membagi data menjadi set train dan test (70:30)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

    st.write("Menampilkan X")
    st.write(X)
    
    st.write("Menampilkan Y")
    st.write(y)
    
