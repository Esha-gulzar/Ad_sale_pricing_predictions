import streamlit as st
import numpy as np
import pandas as pd
import pickle as pkl


# Load the actual pickle file (with .pkl extension)
model = pkl.load(open("Linear_Ad_sales_model.pkl", "rb"))
st.title("Scikit-learn Linear Regression Model")
tv=st.text_input("Enter tv sales:")
Radio=st.text_input("Enter Radio sales:")
newspaper=st.text_input("Enter newspaper sales:")
if st.button("Predict"):
    features=np.array([[tv,Radio,newspaper]] ,dtype=np.float64)
    results=model.predict(features).reshape(1,-1)
    st.write("Predicted sale:",results[0])
    