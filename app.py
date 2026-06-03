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

# Load CSV with UTF-16 encoding for Windows/Excel files
try:
    df = pd.read_csv('transactions.csv', encoding='utf-16')
except UnicodeDecodeError:
    df = pd.read_csv('transactions.csv', encoding='utf-8-sig')
except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

st.success(f"Loaded {len(df)} transactions from CSV")
st.dataframe(df.head())

# Load trained model
try:
    model = joblib.load('fraud_model.pkl')
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Make predictions
X = df.drop('Fraud', axis=1)
y_true = df['Fraud']
y_pred = model.predict(X)

# Metrics
st.subheader("Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.text("Classification Report:")
    st.text(classification_report(y_true, y_pred))

with col2:
    st.text("Confusion Matrix:")
    fig
