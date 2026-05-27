import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Credit Card Fraud Detection Dashboard")
st.markdown("---")

# 1. Generate sample data
np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({
    'Amount': np.random.exponential(80, n_samples).round(2),
    'Hour': np.random.randint(0, 24, n_samples),
    'Fraud': np.random.choice([0, 1], n_samples, p=[0.97, 0.03])
})

# 2. Top metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Fraud Cases", f"{df['Fraud'].sum()}")
col3.metric("Fraud Rate", f"{df['Fraud'].mean()*100:.2f}%")

st.markdown("---")

# 3. Graphs row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Fraud vs Legitimate Count")
    fig, ax = plt.subplots(figsize=(6,4))
    fraud_counts = df['Fraud'].value_counts().sort_index()
    ax.bar(['Legitimate', 'Fraud'], fraud_counts, color=['#2ecc71', '#e74c3c'])
    ax.set_ylabel('Number of Transactions')
    for i, v in enumerate(fraud_counts):
        ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
    st.pyplot(fig)

with col2:
    st.subheader("2. Confusion Matrix")
    y_pred = np.random.choice([0, 1], n_samples, p=[0.975, 0.025])
    cm = confusion_matrix(df['Fraud'], y_pred)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legit', 'Fraud'], 
                yticklabels=['Legit', 'Fraud'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

# 4. Graph row 2
st.subheader("3. Transaction Amount by Fraud Status")
fig, ax = plt.subplots(figsize=(10,4))
sns.boxplot(data=df, x='Fraud', y='Amount', ax=ax, hue='Fraud', 
            palette=['#2ecc71', '#e74c3c'], legend=False)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Legitimate', 'Fraud'])
ax.set_ylabel('Amount ($)')
ax.set_xlabel('')
st.pyplot(fig)