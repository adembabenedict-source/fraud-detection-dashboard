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
# Your file is UTF-16-LE, 1 value per line
df_raw = pd.read_csv('transactions.csv', encoding='utf-16-le', header=None)
st.write(f"Raw shape: {df_raw.shape}")

# First 4 rows are headers: amount, time, is_fraud, transaction_type
headers = df_raw.iloc[0:4, 0].tolist()
st.write("Detected headers:", headers)

# Remaining rows are data, reshape into 4 columns
data_values = df_raw.iloc[4:, 0].to_numpy() # FIX: Arrow -> numpy
num_cols = len(headers)
num_rows = len(data_values) // num_cols

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

st.subheader("Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.text("Classification Report:")
    st.code(classification_report(y_true, y_pred))

with col2:
    st.text("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

st.subheader("Data Distribution")
fig2, ax2 = plt.subplots()
df['Fraud'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_title('Fraud vs Legitimate Transactions')
ax2.set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
st.pyplot(fig2)
