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

st.write(""" 
# WELCOME TO PREDICT HEPATITIS C SYSTEM
""")

st.write("========================================================================================")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Deskripsi Data", "Dataset (Import Data)", "Preprocessing", "Modelling", "Implementasi"])

with tab1:
    st.write("HEPATITIS C DATASET")
    st.write("Dataset ini berisi nilai laboratorium donor darah dan pasien Hepatitis C serta nilai demografi seperti usia")
    st.write("Data ini digunakan untuk memodelkan penyakit Hepatitis C")
    st.write("Fitur - Fitur dalam dataset ini yang akan digunakan ialah sebagai berikut.")
    st.write("1. ALB (Albumin) : Numerik ")
    st.write("2. ALP (Alkaline Phosphatase) : Numerik ")
    st.write("3. ALT (Alanine Transaminase) : Numerik ")
    st.write("4. AST (Aspartate aminotransferase) : Numerik ")
    st.write("5. BIL (Bilirubin) : Numerik ")
    st.write("6. CHE (Cholinesterase) : Numerik ")
    st.write("7. CHOL (Cholesterol) : Numerik ")
    st.write("8. CREA (Creatin) : Numerik ")
    st.write("9. GGT (Gamma-glutamyl transferase) : Numerik ")
    st.write("10. PROT (Protein) : Numerik ")
    st.write("11. Sex : Kategorikal ")
    st.write("12. Age  : Numerik ")
    st.write("13. Category : Kategorikal (0 = Blood Donor, 1 = Suspect Blood Donor, 2 = Hepatitis, 3 = Fibrosis, 4 = Cirrhosis ")
    st.write("Sumber Data : https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset")
    st.write("Sumber Dataset Cleaning : https://raw.githubusercontent.com/Aisyahmsp/datamining/main/hepatitis.csv")
    

with tab2:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)
        
with tab3 : 
    st.write("""# Preprocessing""")
    df[["Age", "Sex", "ALB","ALP","ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]].agg(['min','max'])

    df.Category.value_counts()
    df = df.drop(columns="Unnamed: 0")

    X = df.drop(columns="Category")
    y = df.Category
    "### Membuang fitur yang tidak diperlukan"
    df

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    "### Transformasi Label"
    y

    le.inverse_transform(y)

    labels = pd.get_dummies(df.Category).columns.values.tolist()

    "### Label"
    labels
    st.markdown("# Normalize")

    "### Normalize data"

    dataubah=df.drop(columns="Sex")
    dataubah

    "### Normalize data sex"
    data_sex=df[['Sex']]
    sex = pd.get_dummies(data_sex)
    sex

    dataOlah = pd.concat([sex], axis=1)
    dataHasil = pd.concat([df,dataOlah], axis = 1)

    X = dataHasil.drop(columns=["Sex","Category"])
    y = dataHasil.Category
    "### Normalize data hasil"
    X

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    "### Normalize data transformasi"
    X

    X.shape, y.shape

    le.inverse_transform(y)

    labels = pd.get_dummies(dataHasil.Category).columns.values.tolist()
    
    "### Label"
    labels

    # """## Normalisasi MinMax Scaler"""


    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X

    X.shape, y.shape


with tab4 :
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # from sklearn.feature_extraction.text import CountVectorizer
    # cv = CountVectorizer()
    # X_train = cv.fit_transform(X_train)
    # X_test = cv.fit_transform(X_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")

    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(X_train, y_train)


    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(X_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(X_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)
        
with tab5 :
    st.write("# Implementation")
    Age = st.number_input('Masukkan Umur Pasien (Contoh : 32)')

    # Sex
    sex = st.radio("Sex",('Male', 'Female'))
    if sex == "Male":
        gen_Female = 0
        gen_Male = 1
    else :
        gen_Female = 1
        gen_Male = 0
    
    ALB = st.number_input('Masukkan Hasil Test ALB (Contoh : 46.9000)')
    ALP = st.number_input('Masukkan Hasil Test ALP (Contoh : 70.3000)')
    ALT = st.number_input('Masukkan Hasil Test ALT (Contoh : 7.7000)')
    AST = st.number_input('Masukkan Hasil Test AST (Contoh : 17.8000)')
    BIL = st.number_input('Masukkan Hasil Test BIL (Contoh : 9.6000)')
    CHE = st.number_input('Masukkan Hasil Test CHE (Contoh : 5.8200)')
    CHOL = st.number_input('Masukkan Hasil Test CHOL (Contoh : 3.1900)')
    CREA = st.number_input('Masukkan Hasil Test CREA (Contoh : 106.000)')
    GGT = st.number_input('Masukkan Hasil Test GGT (Contoh : 46.800)')
    PROT = st.number_input('Masukkan Hasil Test PROT (Contoh : 47.1000)')               



    def submit():
        # input
        inputs = np.array([[
            Age,
            gen_Female, gen_Male,
            ALB, ALP, ALT, AST,
            BIL, CHE, CHOL, CREA,
            GGT, PROT
            ]])

        le = joblib.load("le.save")

        if akurasi > skor_akurasi and akurasiii:
            model = joblib.load("nb.joblib")

        elif skor_akurasi > akurasi and akurasiii:
            model = joblib.load("knn.joblib")

        elif akurasiii > skor_akurasi and akurasi:
            model = joblib.load("tree.joblib")

        y_pred3 = model.predict(inputs)
        st.write(f"Berdasarkan data yang Anda masukkan, maka anda diprediksi cenderung : {le.inverse_transform(y_pred3)[0]}")
        st.write("0 = Blood Donor")
        st.write("1 = Suspect Blood Donor")
        st.write("2 = Hepatitis")
        st.write("3 = Fibrosis")
        st.write("4 = Cirrhosis")
    all = st.button("Submit")
    if all :
        st.balloons()
        submit()
