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

# Load CSV - handle UTF-16 + tab separated + transposed
try:
    df = pd.read_csv('transactions.csv', encoding='utf-16', sep=None, engine='python')
except:
    df = pd.read_csv('transactions.csv', encoding='utf-8-sig', sep=None, engine='python')

# Fix if CSV is transposed
if df.shape[1] == 1 and df.columns[0].lower() == 'amount':
    df = df.T
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)

# Standardize column names
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.rename(columns={'is_fraud': 'Fraud', 'transaction_type': 'type'})

st.success(f"Loaded {len(df)} transactions from CSV")
st.dataframe(df.head())

# Load trained model
try:
    model = joblib.load('fraud_model.pkl')
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Make predictions
if 'Fraud' not in df.columns:
    st.error("Column 'Fraud' not found. Available columns: " + str(list(df.columns)))
    st.stop()

X = df.drop('Fraud', axis=1)
y_true = df['Fraud'].astype(int)
y_pred = model.predict(X)

# Metrics
st.subheader("Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.text("Classification Report:")
    st.text(classification_report(y_true, y_pred))

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
