import streamlit as st
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import altair as alt
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# from sklearn.ensemble import BaggingClassifier
# from sklearn.datasets import make_classification
# from sklearn.svm import SVC
# from sklearn import metrics
# from pickle import dump
# import joblib
# from PIL import Image
# from awesome_table import AwesomeTable
# %matplotlib inline
import matplotlib.pyplot as plt
# showWarningOnDirectExecution = False

with st.sidebar:
    selected = option_menu(
        menu_title= "MENU",
        options=["HOME","FITUR","PROJECT"],
    )

if selected == "HOME":
    st.markdown ('<h1 style = "text-align: center;"> STRES </h1>', unsafe_allow_html = True)
    # gambar = Image.open("stress.png")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.image("stress.png",width=300)
    with col3:
        st.write(' ')

    st.write(' ')
    st.markdown ('<p style = "text-align: justify;"> <b> Stres </b> adalah reaksi seseorang baik secara fisik maupun emosional (mental/psikis) apabila ada perubahan dari lingkungan yang mengharuskan seseorang menyesuaikan diri. Stres adalah bagian alami dan penting dari kehidupan, tetapi apabila berat dan berlangsung lama dapat merusak kesehatan kita. </p>', unsafe_allow_html = True)

    st.write(' ')
    st.markdown ('<p style = "text-align: justify;"> Stres pada manusia dapat diklasifikasikan menjadi eustres, neustres, dan distres. Eustress dianggap sebagai stres yang “baik” dan dapat memotivasi seseorang untuk meningkatkan kinerja. Stres netral disebut neustress. Karena tidak membahayakan kesejahteraan seseorang, itu dapat diabaikan. Stres dengan efek negatif pada tubuh manusia disebut kesusahan dan merupakan jenis stres yang penting untuk difokuskan. Tergantung pada karakteristik waktunya, distres diklasifikasikan menjadi stres akut dan kronis. Stres akut adalah tingkat stres yang singkat tetapi intens, sedangkan tingkat intens jangka panjang dianggap sebagai stres kronis. Stres kronis memiliki konsekuensi yang sangat serius pada hidup sehat manusia. Stres meningkatkan ketegangan otot dan menyebabkan gangguan dalam aktivitas fisik sehari-hari.</p>', unsafe_allow_html = True)
    
elif selected == "FITUR":
    st.markdown ('<h1 style = "text-align: center;"> FITUR DATA </h1>', unsafe_allow_html = True)
    st.markdown('<p style = "text-align: justify;">Dalam menentukan nilai yang dilakukan untuk pengklasifikasian sistem ini maka diperlukan beberapa data pada fitur-fitur yang diperlukan, terdapat 3 fitur yang digunakan sebagai parameter pengklasifikasian data, yaitu:</p><ol><b>1. Humidity (Kelembaban keringat)</b></ol><p style="text-align:justify;">Kelembaban yang digunakan yaitu Kelembaban sekresi keringat yang berhubungan dengan sistem saraf pusat, dengan menggunakan <b style ="color:red">Humidity Sensor</b> akan mengetahui kuantitas fisik yang dikeluarkan melalui pori-pori kulit dalam jumlah tertentu sebagai reaksi terhadap panas, latihan fisik dan perubahan emosi. Saat keringat tubuh meningkat, aliran arus antara dua elektroda meningkat membuat tubuh manusia efektif sebagai resistor variabel. Sensor yang mendeteksi kelembapan dapat digunakan untuk memantau tingkat sekresi keringat yang dikendalikan oleh sistem saraf pusat manusia. Memantau jumlah keringat yang dihasilkan dapat membantu menemukan tingkat stres dan gairah subjek yang dipantau. Aktivitas kelenjar keringat sebagai variabel digunakan dalam banyak aplikasi biofeedback seperti deteksi kebohongan, dan pengenalan emosi. Proses berkeringat normal disebut keringat sedangkan gangguan keringat berlebih dikenal sebagai hiperhidrosis dan berhubungan dengan emosional, stres pekerjaan dan sosial. Dalam hal ini digunakan sensor kelembaban untuk mendeteksi sekresi keringat pada telapak tangan.</p> <ol><b>2. Temperature (Suhu tubuh)</b></ol> <p style="text-align:justify">Tingkat suhu adalah tingkat variasi suhu tubuh dalam jumlah waktu tertentu. Secara umum, sensor suhu dapat diklasifikasikan dalam 2 jenis: Sensor suhu kontak yang mengukur suhu saat diletakkan di tubuh dan sensor non-kontak yang mengukur radiasi infra merah atau optik yang diterima dari area tubuh mana pun. Untuk mengukur suhu tubuh mnggunakan <b style ="color:red">sensor suhu kontak</b> yang dapat memantau laju variasi suhu tubuh.</p> <ol><b>3. Step count (Jumlah langkah)</b></ol><p style="text-align:justify">Dengan melakukan sebuah aktivitas maka dapat mempengaruhi stres pada manusia, untuk mengukur laju perubahan kecepatan suatu benda digunakan <b style ="color:red">sensor akselerometer</b>. Yang terdiri dari tiga akselerometer terpisah yang dipasang secara ortogonal pada sistem 3 sumbu fisik (x,y, danz). Gaya yang menyebabkan percepatan bisa statis atau dinamis.yang diukur dalam meter per detik persegi (m/s2)</p>',unsafe_allow_html=True)

else:
    st.markdown ('<h1 style = "text-align: center;"> CEK TINGKAT STRES</h1>', unsafe_allow_html = True)
    st.write("Oleh | FIQRY WAHYU DIKY W | 200411100125")
    data, preprocessing, modelling, evaluasi, implementasi = st.tabs(["Data","Preprocessing","Modelling","Evaluasi","Implementasi"])
#=============================================================================
    with data:
        dataset, keterangan = st.tabs(["Dataset", "Keterangan"])
        with dataset:
            st.write("# Data")
            data = pd.read_csv("Stress-Lysis.csv")
            data
        with keterangan:
            st.write("Berikut beberapa keterangan yang ada dalam dataset:")
            st.info("#### Tipe data")
            data.dtypes
            #===================================
            st.info("#### Nilai min-maks data")
            col1,col2 = st.columns(2)
            with col1:
                st.write("##### Nilai minimum")
                st.write(data.min())
            with col2:
                st.write("##### Nilai maksimal")
                st.write(data.max())
            #===================================
            st.info("##### Data kosong")
            st.write(data.isnull().sum())

#======================= Preprocessing =================================           
    with preprocessing:
        st.write("# Preprocessing")
        st.write("Sebelum melakukan modelling data harus diprocessing dahulu agar komputasi sistem lebih mudah dibaca, akurasi data, kelengkapan, konsistensi, ketepatan waktu, tepercaya, serta dapat diinterpretasi dengan baik")
        col1, col2 = st.columns(2)
        with col1:
            st.info("Data sebelum Normalisasi")
            st.write(data[['Humidity','Temperature','Step count']])
        with col2:
            st.info("Data Normalisasi")
            scaler  = MinMaxScaler()
            scaled  = scaler.fit_transform(data[['Humidity','Temperature','Step count']])
            kolom_normalisasi   = ["Humaditiy","Temperature","Step count"]
            data_normalisasi    = pd.DataFrame(scaled,columns=kolom_normalisasi)
            st.write (data_normalisasi)
#=========================== Modeling ===============================
    with modelling:
        st.write("# Modeling")
        st.write("Dalam sistem ini menggunakan 3 modeling yaitu KNN, Naive-Bayes, dan Decission Tree")
        knn_cekbox          = st.checkbox("KNN")
        bayes_gaussian_cekbox  = st.checkbox("Naive-Bayes Gaussian")
        decission3_cekbox     = st.checkbox("Decission Tree")
#=========================== Spliting data ======================================
        X   = data_normalisasi.iloc[:,:4]
        # X
        Y  = data.iloc[:,-1]
        # Y
        X_train, X_test, Y_train, Y_test    = train_test_split(X,Y, test_size=0.3, random_state=0)
        # st.write(X_train)
        # st.write(X_test)
        # st.write(Y_train)
        # st.write(Y_test)
#============================ Model =================================
    #===================== KNN =======================
        k_range = range(1,51)
        scores = {}
        scores_list = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
            y_pred_knn = knn.predict(X_test)
            # scores[k] = metrics.accuracy_score(Y_test,y_pred_knn)
            # scores_list.append(metrics.accuracy_score(Y_test,y_pred_knn))
        knn_accuracy = round(100 * accuracy_score(Y_test, y_pred_knn), 2)
        knn_eval = classification_report(Y_test, y_pred_knn,output_dict = True)
        knn_eval_df = pd.DataFrame(knn_eval).transpose()
        # scoress = st.pd.dataframe(scores)

    #===================== Bayes Gaussian =============
        gaussian    = GaussianNB()
        gaussian.fit(X_train,Y_train)
        y_pred_gaussian   =  gaussian.predict(X_test)
        gauss_accuracy  = round(100*accuracy_score(Y_test, y_pred_gaussian),2)
        gaussian_eval = classification_report(Y_test, y_pred_gaussian,output_dict = True)
        gaussian_eval_df = pd.DataFrame(gaussian_eval).transpose()

    #===================== Decission tree =============
        decission3  = DecisionTreeClassifier(criterion="gini")
        decission3.fit(X_train,Y_train)
        y_pred_decission3 = decission3.predict(X_test)
        decission3_accuracy = round(100*accuracy_score(Y_test, y_pred_decission3),2)
        decission3_eval = classification_report(Y_test, y_pred_decission3,output_dict = True)
        decission3_eval_df = pd.DataFrame(decission3_eval).transpose()

        # decission3_accuracy.dtype
        st.markdown("---")
    
    #===================== Cek Box ====================
        if knn_cekbox:
            st.write("##### KNN")
            st.warning("Dengan menggunakan metode KNN yang menggunakan nilai K = 1-50 didapatkan akurasi  sebesar:")
            # st.warning(knn_accuracy)
            st.warning(f"akurasi  =  {knn_accuracy}%")
            st.markdown("---")

        if bayes_gaussian_cekbox:
            st.write("##### Naive Bayes Gausssian")
            st.info("Dengan menggunakan metode Bayes Gaussian didapatkan hasil akurasi sebesar:")
            st.info(f"Akurasi = {gauss_accuracy}%")
            st.markdown("---")

        if decission3_cekbox:
            st.write("##### Decission Tree")
            st.success("Dengan menggunakan metode Decission tree didapatkan hasil akurasi sebesar:")
            st.success(f"Akurasi = {decission3_accuracy}%")

#=========================== Evaluasi ==================================
    with evaluasi:
        st.write("# Evaluasi")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.warning("Model KNN")
            st.write(knn_eval_df)
        with col2:
            st.info("Model Naive-Bayes Gaussian")
            st.write(gaussian_eval_df )
        with col3:
            st.success("Model Decission Tree")
            st.write(decission3_eval_df)
    #===================== Grafik Visual ====================   
        st.markdown("---")
        st.write("##### Visualisasi Model")
        visual = pd.DataFrame({'Akurasi': [knn_accuracy,gauss_accuracy,decission3_accuracy],'Model':['KNN','Naive-Bayes Gaussian','Decission Tree']})
        chart = alt.Chart(visual).mark_bar().encode(
            y = "Akurasi",
            x = "Model",
        )
        st.altair_chart(chart, use_container_width=True)

#=========================== Implementasi ===============================
    with implementasi:
        st.write("# Implementasi")
        st.info("Dalam melakukan pengecekan tingkat stres harus menggunakan 3 fitur data yang didapatkan dari melakukan beberapa aktivitas dan diukur menggunakan sebuah sensor. Aktivitas:")
        col1,col2,col3 = st.columns(3)
        with col1:
            st.write("")
        with col2:
            st.image("aktivitas.png",use_column_width="auto")
        with col3:
            st.write("")
        st.markdown("---")
        st.write("##### Input fitur")
        name = st.text_input("Masukkan nama anda")
        col1,col2,col3 = st.columns(3)
        with col1:
            humidity_mean = st.number_input("Masukkan Rata-rata Kelembaban (10.00 - 30.00)", min_value=10.00, max_value=30.00)
        with col2:
            temperature_mean = st.number_input("Masukkan rata-rata Suhu (79.00 - 99.00) Fahrenheit", min_value=79.00, max_value=99.00)
        with col3:
            step_count_mean = st.number_input("Masukkan rata-rata hitungan langkah (0.00 - 200.00)", min_value=0.00, max_value=200.00)

        cek_hasil = st.button("Cek Prediksi")
#============================ Mengambil akurasi tertinggi ===========================
        if knn_accuracy > gauss_accuracy and knn_accuracy > decission3_accuracy:
            use_model = knn
            metode = "KNN"
        elif gauss_accuracy > knn_accuracy and gauss_accuracy > decission3_accuracy:
            use_model = gaussian
            metode = "Naive-Bayes Gaussian"
        else:
            use_model = decission3
            metode = "Decission Tree"
#============================ Normalisasi inputan =============================
        inputan = [[humidity_mean, temperature_mean, step_count_mean]]
        inputan_norm = scaler.transform(inputan)
        # inputan
        # inputan_norm
        FIRST_IDX = 0
        if cek_hasil:
            hasil_prediksi = use_model.predict(inputan_norm)[FIRST_IDX]
            if hasil_prediksi == 0:
                st.success(f"{name} Terdeteksi tingkat stress tergolong Rendah, dengan tingkat = {hasil_prediksi} Berdasarkan metode {metode}")
            elif hasil_prediksi == 1:
                st.warning(f"{name} Terdeteksi tingkat stress tergolong Normal, dengan tingkat = {hasil_prediksi} Berdasarkan metode {metode}")
            else:
                st.error(f"{name} Terdeteksi tingkat stress tergolong Tinggi, dengan tingkat = {hasil_prediksi} Berdasarkan metode {metode}")
