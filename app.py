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
df_raw = pd.read_csv('transactions.csv', encoding='utf-16-le', header=None)
st.write(f"Raw shape: {df_raw.shape}")

headers = df_raw.iloc[0:4, 0].tolist()
data_values = df_raw.iloc[4:, 0].to_numpy()
num_cols = len(headers)
num_rows = len(data_values) // num_cols

df = pd.DataFrame(data_values.reshape(num_rows, num_cols), columns=headers)
st.write(f"Fixed shape: {df.shape}")

df.columns = df.columns.astype(str).str.lower().str.strip().str.replace(' ', '_')
df = df.rename(columns={'is_fraud': 'Fraud', 'transaction_type': 'type', 'fraud': 'Fraud'})

st.success(f"Loaded {len(df)} transactions")
st.dataframe(df.astype(str))

# Load model
model = joblib.load('fraud_model.pkl')

if 'Fraud' not in df.columns:
    st.error(f"Column 'Fraud' not found. Got: {list(df.columns)}")
    st.stop()

# --- THIS IS THE FIX ---
X = df.drop('Fraud', axis=1)

# 1. Handle categorical 'type' column
if 'type' in X.columns:
    X = pd.get_dummies(X, columns=['type'], drop_first=False)

# 2. Convert everything to numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(0)

# 3. Force X to match the exact columns the model was trained on
if hasattr(model, 'feature_names_in_'):
    # Add any missing columns as 0
    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = 0
    # Drop any extra columns and reorder
    X = X[model.feature_names_in_]
    st.info(f"Aligned features to model. Using: {list(X.columns)}")
else:
    st.warning("Model doesn't store feature names. Using first N columns.")
    X = X.iloc[:, :model.n_features_in_]
# --- END FIX ---

y_true = df['Fraud'].astype(int)
y_pred = model.predict(X)

st.subheader("Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.text("Classification Report:")
    st.code(classification_report(y_true, y_pred,
