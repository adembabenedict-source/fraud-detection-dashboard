import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import os

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection Dashboard")

st.write("Checking files...")
if not os.path.exists('transactions.csv'):
    st.error("transactions.csv not found")
    st.stop()

# Show raw file content to debug
st.subheader("Raw file preview")
with open('transactions.csv', 'rb') as f:
    raw_bytes = f.read(500)
    st.code(f"First 500 bytes: {raw_bytes}")
    
with open('transactions.csv', 'r', encoding='utf-16-le', errors='replace') as f:
    lines = [f.readline() for _ in range(5)]
    st.code("First 5 lines with utf-16-le:\n" + ''.join(lines))

# Try loading with explicit tab separator
st.subheader("Loading CSV")
try:
    df = pd.read_csv('transactions.csv', encoding='utf-16-le', sep='\t', engine='python')
    st.write("Success: utf-16-le + tab")
except Exception as e:
    st.error(f"Load failed: {e}")
    st.stop()

st.write(f"Shape: {df.shape}")
st.write("Columns:", list(df.columns))

if df.empty:
    st.error("DataFrame is empty. Check your CSV file - it might have no data rows.")
    st.stop()

# Fix transpose if needed
if df.shape[1] == 1 and len(str(df.columns[0])) < 5:
    st.write("Detected transposed CSV, fixing...")
    df = df.T
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)

# Standardize
df.columns = df.columns.astype(str).str.lower().str.strip().str.replace(' ', '_')
df = df.rename(columns={'is_fraud': 'Fraud', 'transaction_type': 'type', 'fraud': 'Fraud'})

st.success(f"Loaded {len(df)} transactions")
# Convert to string to avoid pyarrow errors
st.dataframe(df.astype(str).head())

# Load model
if not os.path.exists('fraud_model.pkl'):
    st.error("fraud_model.pkl not found")
    st.stop()
    
model = joblib.load('fraud_model.pkl')

if 'Fraud' not in df.columns:
    st.error(f"Column 'Fraud' not found. Got: {list(df.columns)}")
    st.stop()

X = df.drop('Fraud', axis=1)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(0)

y_true = df['Fraud'].astype(int)
y_pred = model.predict(X)

st.subheader("Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.text("Classification Report:")
    st.code(classification_report(y_true, y_pred))
with col2:
    st.text("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
