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

if not os.path.exists('transactions.csv'):
    st.error("transactions.csv not found in repo")
    st.stop()
if not os.path.exists('fraud_model.pkl'):
    st.error("fraud_model.pkl not found in repo")
    st.stop()

st.subheader("Loading CSV")
# Read as single column - no tabs in this file
df_raw = pd.read_csv('transactions.csv', encoding='utf-16-le', header=None)
st.write(f"Raw shape: {df_raw.shape}")

# First 4 rows are the column names
headers = df_raw.iloc[0:4, 0].tolist()
st.write("Detected headers:", headers)

# Remaining rows are data, reshape into 4 columns
data_values = df_raw.iloc[4:, 0].values
num_cols = len(headers) # Should be 4
num_rows = len(data_values) // num_cols

# Reshape the flat list into rows x cols
df = pd.DataFrame(data_values.reshape(num_rows, num_cols), columns=headers)
st.write(f"Fixed shape: {df.shape}")

# Standardize column names
df.columns = df.columns.astype(str).str.lower().str.strip().str.replace(' ', '_')
df = df.rename(columns={'is_fraud': 'Fraud', 'transaction_type': 'type', 'fraud': 'Fraud'})

st.success(f"Loaded {len(df)} transactions")
st.write("Columns:", list(df.columns))
st.dataframe(df.astype(str).head())

# Load model
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

st.subheader("Model
