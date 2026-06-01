import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection Dashboard")
st.write("Loading real transaction data...")

# Load CSV with encoding fallback for Windows/Excel files
try:
    df = pd.read_csv("transactions.csv", encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv("transactions.csv", encoding="latin1")
except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

st.success(f"Loaded {len(df)} transactions from CSV")
st.dataframe(df.head())

# Load trained model
try:
    model = joblib.load("fraud_model.pkl")
    st.write("Model loaded successfully")
except FileNotFoundError:
    st.error("fraud_model.pkl not found. Push it to GitHub.")
    st.stop()

# Show basic stats
st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", len(df))
col2.metric("Fraud Cases", int(df['Fraud'].sum()))
col3.metric("Fraud Rate", f"{df['Fraud'].mean()*100:.2f}%")

# Plot fraud distribution
st.subheader("Fraud vs Legitimate")
fig, ax = plt.subplots()
sns.countplot(data=df, x='Fraud', ax=ax)
ax.set_xticklabels(['Legitimate', 'Fraud'])
st.pyplot(fig)