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

# Check files exist
if not os.path.exists('transactions.csv'):
    st.error("transactions.csv not found in repo")
    st.stop()
if not os.path.exists('fraud_model.pkl'):
    st.error("fraud_model.pkl not found in repo")
    st.stop()

st.subheader("Loading CSV")
# Step 1: Read transposed Excel UTF-16 file with header=None
df_raw = pd.read_csv('transactions.csv', encoding='utf-16-le', sep='\t', header=None, engine='python')
st.write(f"Raw shape: {df_raw.shape}")

# Step 2: Transpose - your CSV is sideways
df = df_raw.T
st.write("Transposed CSV detected - fixing...")

# Step 3: Set first row as column headers
df.columns = df.iloc[0]
df = df.drop(df.index[0]).reset_index(drop=True)

st.write(f"Fixed shape: {df.shape}")

# Step 4: Standardize column names
df.columns = df.columns.astype(str).str.lower().str.strip().str.replace(' ', '_')
df = df.rename(columns={'is_fraud': 'Fraud', 'transaction_type': 'type', 'fraud': 'Fraud'})

st.success(f"Loaded {len(df)} transactions")
st.write("Columns:", list(df.columns))
st.dataframe(df.astype(str).head())

# Load model
model = joblib.load('fraud_model.pkl')

# Make predictions
if 'Fraud' not in df.columns:
    st.error(f"Column 'Fraud' not found. Got: {list(df.columns)}")
    st.stop()

X = df.drop('Fraud', axis=1)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(0)

y_true = df['Fraud'].astype(int)
y_pred = model.predict(X)

# Metrics
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

# Fraud distribution
st.subheader("Data Distribution")
fig2, ax2 = plt.subplots()
df['Fraud'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_title('Fraud vs Legitimate Transactions')
ax2.set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
st.pyplot(fig2)
