import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib
from sklearn.preprocessing import StandardScaler
import joblib
from io import StringIO, BytesIO
import urllib.request
import time
import os,sys
from scipy import stats



st.set_page_config(page_title="Diah Kamalia")

st.title("Aplikasi Data Mining - Water Quality")
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
        df["ph"].fillna(0, inplace=True)
        df["Hardness"].fillna(0, inplace=True)
        df["Solids"].fillna(0, inplace=True)
        df["Chloramines"].fillna(0, inplace=True)
        df["Sulfate"].fillna(0, inplace=True)
        df["Conductivity"].fillna(0, inplace=True)
        df["Organic_carbon"].fillna(0, inplace=True)
        df["Trihalomethanes"].fillna(0, inplace=True)
        df["Turbidity"].fillna(0, inplace=True)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)
        
        
with preprocessing : 
    st.write("""# Preprocessing""")
    st.write("""
    > Preprocessing data adalah proses menyiapkan data mentah dan membuatnya cocok untuk model pembelajaran mesin. Ini adalah langkah pertama dan penting saat membuat model pembelajaran mesin. Saat membuat proyek pembelajaran mesin, kami tidak selalu menemukan data yang bersih dan terformat.
    """)
    st.write("### Formula")
    st.latex(r'''
    X = \frac{X_i - X_{min}}{X_{max} - X_{min}}
    ''')
    st.write("Data Before Normalization")
    st.dataframe(df)

    label = df["Potability"]

    st.write("## Drop Column")
    st.write("Dropping 'Potability' Table")
    X = df.drop(columns=["Potability"])

    st.dataframe(X)

    st.write(" ## Normalisasi ")
    st.write(""" > ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity """)
    st.write("#### Data After Normalization ")
    label=df["Potability"]
    data_for_minmax_scaler=pd.DataFrame(df, columns = ['ph', 'Hardness', 'Solids', 'Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity'])
    data_for_minmax_scaler.to_numpy()
    scaler = MinMaxScaler()
    hasil_minmax=scaler.fit_transform(data_for_minmax_scaler)
    hasil_minmax = pd.DataFrame(hasil_minmax,columns = ['ph', 'Hardness', 'Solids', 'Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity'])
    st.dataframe(hasil_minmax)

    X = hasil_minmax
    X=X.join(label) 
    st.write("## New Dataframe")
    st.dataframe(X)
    st.write("## Counting Data")
    st.write("- Drop Potability column from dataframe")
    st.write("- Split Data Training and Data Test")
    st.write(""" ### Spliting Data
            - Data Training - X_train 
            - Data Test - X_test 
            - Data Training (Class) - y_train
            - Data Test (Class) - y_test
            """)
    
    # memisahkan data Potability
    X = hasil_minmax
    y=pd.DataFrame(df, columns=["Potability"])
    # membagi data menjadi set train dan test (70:30)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)

    st.write("Showing X")
    st.write(X)
    
    st.write("Showing Y")
    st.write(y)

with modelling : 
    st.write("""# Modelling""")
    knn,gaussian,decision,random = st.tabs(["K-Nearest Neighbor", "Gaussian Naive Bayes", "Decision Tree", "Random Forest"])
    with knn:
        st.write("## K-Nearest Neighbor")
        st.write("""
        > Algortima K-Nearest Neighbor (KNN) adalah merupakan sebuah metode untuk melakukan klasifikasi terhadap obyek baru berdasarkan (K) tetangga terdekatnya. Metode pencarian jarak yang digunakan adalah Euclidean Distance yaitu perhitungan jarak terdekat.
        """)
        st.write("### Formula")
        st.latex(r'''
        d(p,q) = d(p,q) = \sqrt{(q_1 - p_1)^2 + (q_2 - p_2)^2 + . . . + (q_n - p_n)^2}
                        = \sqrt{\displaystyle\sum_{i=1}^n (q_i - p_i)^2}
        ''')
        st.write("### Step of the K-Nearest Neighbor Algorithm")
        st.write("""
        1. Menentukan parameter K (jumlah tetangga paling dekat)

        2. Menghitung kuadrat jarak euclidian (euclidean distance) masing-masing obyek terhadap data sampel yang diberikan

        3. Mengurutkan objek-objek tersebut ke dalam kelompok yang mempunyai jarak euclidean terkecil

        4. Mengumpulkan kategori Y (klasifikasi nearest neighbor)

        5. Dengan menggunakan kategori mayoritas,maka dapat diprediksikan nilai query instance yang telah dihitung
        """)
        # Inisialisasi K-NN
        my_param_grid = {'n_neighbors':[2,3,5,7], 'weights': ['distance', 'uniform']}
        GridSearchCV(estimator=KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
        knn = GridSearchCV(KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
        knn.fit(X_train, y_train)

        pred_test = knn.predict(X_test)

        modknn = f'Water Potability accuracy of K-Nearest Neighbour model is : **{accuracy_score(y_test, pred_test) * 100 :.2f} %**'

        filenameModelKnn = 'KNN.pkl'
        joblib.dump(knn, filenameModelKnn)
        st.header("Accuracy Result")
        st.write(modknn)

    with gaussian:
        st.write("## Gaussian Naive Bayes")
        st.write("""
        > Naive Bayes adalah algoritma machine learning untuk masalah klasifikasi. Ini didasarkan pada teorema probabilitas Bayes. Hal ini digunakan untuk klasifikasi teks yang melibatkan set data pelatihan dimensi tinggi. Beberapa contohnya adalah penyaringan spam, analisis sentimental, dan klasifikasi artikel berita.
        """)
        st.write("### Formula")
        st.latex(r'''
        P(C_k | x) = \frac{P(C_k) P(x|C_k)}{P(x)}
        ''')
        # Inisialisasi Gaussian
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)

        modgnb = f'Water Potability accuracy of Gaussian Naive Bayes model is : **{accuracy_score(y_test, y_pred) * 100 :.2f} %**'

        filenameModelGau = 'Gaussian.pkl'
        joblib.dump(gnb, filenameModelGau)
        st.header("Accuracy Result")
        st.write(modgnb)

    with decision:
        st.write("## Decision Tree")
        st.write("""
        > Decision tree merupakan alat pendukung keputusan dengan struktur seperti pohon yang memodelkan kemungkinan hasil, biaya sumber daya, utilitas, dan kemungkinan konsekuensi. Konsepnya adalah dengan cara menyajikan algoritma dengan pernyataan bersyarat yang meliputi cabang untuk mewakili langkah-langkah pengambilan keputusan yang dapat mengarah pada hasil yang menguntungkan.
        """)
        st.write("### Define Decision Tree Roots")
        st.write("""
        > Akar akan diambil dari atribut yang terpilih, dengan cara menghitung nilai gain dari masing – masing atribut. Nilai gain yang paling tinggi akan menjadi akar pertama. Sebelum menghitung niali gain dari atribut, harus menghitung nilai entropy terlebih dahulu
        """)
        st.write("### Formula Entropy")
        st.latex(r'''
        Entropy\left(\LARGE{D_1}\right) = - \displaystyle\sum_{i=1}^m p_i log_2 p_i
        ''')
        st.write("### Formula Gain (D1)")
        st.latex(r'''
        Gain(E_{new}) = E_{initial} - E_{new}
        ''')
        # Inisialisasi Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        d3 = DecisionTreeClassifier()
        d3.fit(X_train, y_train)

        y_pred = d3.predict(X_test)

        moddt = f'Water Potability accuracy of Decision Tree model is : **{accuracy_score(y_test, y_pred) * 100 :.2f} %**'

        filenameModelDT = 'DecisionTree.pkl'
        joblib.dump(d3, filenameModelDT)
        st.header("Accuracy Result")
        st.write(moddt)

    with random:
        st.write("## Random Forest")
        st.write(""" 
        > Random forest  adalah kombinasi dari  masing – masing tree yang baik kemudian dikombinasikan  ke dalam satu model. Random Forest bergantung pada sebuah nilai vector random dengan distribusi yang sama pada semua pohon yang masing masing decision tree memiliki kedalaman yang maksimal. Random forest adalah classifier yang terdiri dari classifier yang berbentuk pohon di mana random vector diditribusikan secara independen dan masing masing tree pada sebuah unit kan memilih class yang paling popular pada input x.
        """)
        st.write("### Formula")
        st.latex(r'''
        P_{X,Y} (P_{\theta}(h(X, \theta) = Y - \max\limits_{j \not = Y} P_{\theta}(h(X, \theta) = j) < 0)
        ''')
        # Inisialisasi Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=14, max_depth=2, random_state=0)
        clf = clf.fit(X_train, y_train)

        y_test_pred = clf.predict(X_test)
        modrf = f'Water Potability accuracy of Random Forest model is : **{accuracy_score(y_test, y_test_pred) * 100 :.2f} %**'

        filenameModelrmf = 'RandomForest.pkl'
        joblib.dump(d3, filenameModelrmf)
        st.header("Accuracy Result")
        st.write(modrf)

with implementation:
    st.write("# Implementation")
    st.write("### Input Data :")
    ph = st.number_input("pH")
    Hardness = st.number_input("Hardness")
    Solids = st.number_input("Solids")
    Chloramines = st.number_input("Chloramines")
    Sulfate = st.number_input("Sulfate")
    Conductivity = st.number_input("Conductivity")
    Organic_carbon = st.number_input("Organic Carbon")
    Trihalomethanes = st.number_input("Trihalomethanes")
    Turbidity = st.number_input("Turbidity")
    def submit():
        a = np.array([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
        test_data = np.array(a).reshape(1, -1)
        test_data = pd.DataFrame(test_data)
        scaler = MinMaxScaler()
        import joblib
        filename = "norm.sav"
        joblib.dump(scaler, filename) 

        scaler = joblib.load(filename)
        test_d = scaler.fit_transform(test_data)

        # load
        knn = joblib.load(filenameModelKnn)
        pred = knn.predict(test_d)

        # load 
        gnb = joblib.load(filenameModelGau)
        pred = gnb.predict(test_d)

        # load
        d3 = joblib.load(filenameModelDT)
        pred = d3.predict(test_d)

        # load
        km = joblib.load(filenameModelrmf)
        pred = km.predict(test_d)


        # button
        st.header("Data Input")
        st.write("Table of input results from data testing :")
        labelmodel=['ph', 'Hardness', 'Solids', 'Chloramines','Sulfate','Conductivity','Organic Carbon','Trihalomethanes','Turbidity']
        model = pd.DataFrame(a, columns = labelmodel)
        st.dataframe(model)

        st.header("Classification Result")
        near, naive, tree, forest = st.tabs(["K-Nearest Neighbour", "Naive Bayes Gaussian", "Decision Tree", "Random Forest"])
        
        with near:
            st.subheader("Model K-Nearest Neighbour")
            pred = knn.predict(test_d)
            if pred[0]== 0 :
                st.write("I'm Sorry, the water you tested is **Not Potable**")
            elif pred[0]== 1 :
                st.write("Good news, the water you tested is **Potable**")

        with naive:
            st.subheader("Model Naive Bayes Gausian")
            pred = gnb.predict(test_d)
            if pred[0]== 0 :
                st.write("I'm Sorry, the water you tested is **Not Potable**")
            elif pred[0]== 1 :
                st.write("Good news, the water you tested is **Potable**")
                
        with tree:
            st.subheader("Model Decision Tree")
            pred = d3.predict(test_d)
            if pred[0]== 0 :
                st.write("I'm Sorry, the water you tested is **Not Potable**")
            elif pred[0]== 1 :
                st.write("Good news, the water you tested is **Potable**")

        with forest:
            st.subheader("Model Random Forest")
            pred = km.predict(test_d)
            if pred[0]== 0 :
                st.write("I'm Sorry, the water you tested is **Not Potable**")
            elif pred[0]== 1 :
                st.write("Good news, the water you tested is **Potable**")

    submitted = st.button("Submit")
    if submitted:
        submit()
