import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection Dashboard")
st.write("Loading real transaction data...")

# Load your real CSV instead of np.random
df = pd.read_csv("transactions.csv")
st.success(f"Loaded {len(df)} transactions from CSV")
st.dataframe(df.head())

# Load your trained model
model = joblib.load("fraud_model.pkl")
st.write("Model loaded successfully")