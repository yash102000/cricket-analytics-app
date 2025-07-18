# app.py
import streamlit as st
from eda import run_eda
from prediction import run_prediction

st.set_page_config(page_title="Cricket Analytics Dashboard", layout="wide")

st.title("🏏 Cricket Data Analytics App")

menu = ["Exploratory Data Analysis", "Match Outcome Prediction"]
choice = st.sidebar.selectbox("Select Feature", menu)

if choice == "Exploratory Data Analysis":
    run_eda()
elif choice == "Match Outcome Prediction":
    run_prediction()
