import pickle
import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageOps
import pytesseract
import numpy as np
import requests
import io
import string

st.set_page_config(
    page_title="ELINA",
    page_icon="‼️",
    menu_items={
        'Get Help': 'https://www.google.com/',
        'Report a bug': "https://www.google.com/",
        'About': "# Hai, ini adalah Final Projcet group 3 FTDS Hacktiv8!"
           }
)


#st.title("Instagram Fake Account Detector")
st.markdown("<h1 style='text-align: center; color: black;'>ELINA</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>FRAUD DETECTION APPLICATION</h1>", unsafe_allow_html=True)

#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

upload = st.file_uploader("Silahkan upload KTP anda ...", type=["jpg","png","jpeg"])
if upload is not None:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(upload, caption='Uploaded Picture', use_column_width=True)
    #convert image to grayscale
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    #apply thresholding to get better results
    ret, thresh = cv2.threshold(gray, 115, 255, cv2.THRESH_TRUNC)

    result = pytesseract.image_to_string(thresh, lang="ind")
    for word in result.split("\n"):
        if "”—" in word:
            word = word.replace("”—", ":")
  
        if "NIK" in word:
            nik_char = word.split(" ")
            if "D" in word:
                word = word.replace("D", "0")
            if "?" in word:
                word = word.replace("?", "7") 

    def Convert(text):
        text_list = list(text.split(" "))
        return text_list
    def nik_detect(l):
        for i in range(len(l)):
            if l[i] == 'NIK':
                nik = l[i+2]
                return nik

    result = result.replace('\n', ' ')
    ktp1 = Convert(result)
    nik1 = nik_detect(ktp1)

# database dummy
data  = [['3474140209790001', 111, 8, 'fraud_Haley, Batz and Auer', 'health_fitness', 'Sales professional, IT', 43, 34, 0],
['3474140209210001', 111, 8, 'fraud_Haley, Batz and Auer', 'health_fitness', 'Sales professional, IT', 43, 34, 0],
['2105042604959001', 780, 12, 'fraud_Metz-Boehm', 'shopping_pos', 'Furniture designer', 27, 15, 1],
['2105042504950001', 250, 98, 'fraud_Metz-Boehm', 'shopping_pos', 'Furniture designer', 27, 15, 0]]
df  = pd.DataFrame(data, columns=['id', 'amt', 'hour_of_day', 'merchant', 'category', 'job', 'age', 'distance','is_fraud'])

if st.button("Cek NIK"):
    st.write("NIK : ", nik1)
    print(nik1 in df['id'].values)
    type(nik1)
    if nik1 in df['id'].values:
        if len(df[(df['id']==nik1) & (df['is_fraud'] == 1)].id.values)>0:
            st.write("NIK nasabah terdaftar di database dan statusnya berbahaya, karena adanya rekam jejak penipuan.")
        else:
            st.write("NIK nasabah terdaftar di database dan statusnya normal, transaksi bisa dilanjutkan.")
    else:
        st.write("NIK anda tidak terdaftar atau anda bukan nasabah kami, yuk nabung disini dan jadilah nasabah kami. Dapatkan bermacam promo menarik dan daftarkan kartu kredit anda")

nik = st.text_input("Jika anda NIK tidak terbaca silahkan masukan NIK secara manual :")
if st.button("Check NIK"):
    if nik in df['id'].values:
        if len(df[(df['id']==nik) & (df['is_fraud'] == 1)].id.values)>0:
            st.write("NIK nasabah terdaftar di database dan statusnya berbahaya, karena adanya rekam jejak penipuan.")
        else:
            st.write("NIK nasabah terdaftar di database dan statusnya normal, transaksi bisa dilanjutkan.")
    else:
        st.write("NIK anda tidak terdaftar atau anda bukan nasabah kami, yuk nabung di sini dan jadilah nasabah kami. Dapatkan bermacam promo menarik dan daftarkan kartu kredit anda.")


# loading the trained model
pkl_model = open('knn_test_3.pkl', 'rb') 
knn = pickle.load(pkl_model)

#loading the pipeline
pkl_pipe = open('pipe_test_3.pkl', 'rb') 
pipe = pickle.load(pkl_pipe)

@st.cache()

def prediction(amt, hour_of_day, merchant, category, job, age, distance):
    daftar_kolom  = ['amt', 'hour_of_day', 'merchant', 'category', 'job', 'age', 'distance']
    dummy  = pd.DataFrame(columns=daftar_kolom)
    dummy.loc[len(dummy.index)] = [amt, hour_of_day, merchant, category, job, age, distance]
    data_pipe = pipe.transform(dummy)
    pred = knn.predict(data_pipe)
    if pred == 1:
        res = 'tolak'        
    else:
        res = 'terima'
    return res


def main():       

    amt = st.number_input('Amount of Transaction') 
    hour_of_day = st.slider('Hour of Transaction', 0, 23, 9)
    merchant = st.selectbox('Merchant Name',('fraud_Haley, Batz and Auer', 'fraud_Waelchi Inc', 'fraud_Jast and Sons', 'fraud_Metz-Boehm'))
    category = st.selectbox('Category of Merchant',('health_fitness', 'kids_pets', 'food_dining', 'shopping_pos', 'home', 'entertainment', 'misc_net'))    
    job = st.selectbox('Job of Credit Card Holder',('Mechanical engineer', 'Sales professional, IT', 'Librarian, public', 'Set designer', 'Furniture designer'))
    age = st.slider('Age of Credit Card Holder (y)', 0, 100, 40)
    distance = st.slider('Distance from Merchant to Credit Card Holder (km)', 0, 150, 60)
    
    result =""
      
    
    if st.button("Predict"): 
        result = prediction(amt, hour_of_day, merchant, category, job, age, distance) 
        st.success('Transaksi nasabah ini di {}'.format(result))
        
        
if __name__=='__main__': 
    main()