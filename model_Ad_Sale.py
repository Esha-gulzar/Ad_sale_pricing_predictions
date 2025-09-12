import streamlit as st
import numpy as np
import pandas as pd
import pickle as pkl

model=pkl.load(open("C:\\Users\A I TECH\Desktop\\folder\\Linear_Ad_sales_model","rb"))

st.title("Scikit-learn Linear Regression Model")
tv=st.text_input("Enter tv sales:")
Radio=st.text_input("Enter Radio sales:")
newspaper=st.text_input("Enter newspaper sales:")
if st.button("Predict"):
    features=np.array([[tv,Radio,newspaper]] ,dtype=np.float64)
    results=model.predict(features).reshape(1,-1)
    st.write("Predicted sale:",results[0])
    