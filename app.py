import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
import os
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

MODEL_PATH = os.getenv("MODEL_PATH", "fraud_model.pkl")
MAX_UPLOAD_SIZE_MB = 200

@st.cache_resource
def load_model(path):
    if not Path(path).exists():
        st.error(f"Model file not found at {path}")
        st.stop()
    logger.info(f"Loading model from {path}")
    return joblib.load(path)

def validate_and_process_csv(uploaded_file):
    df_raw = pd.read_csv(uploaded_file, encoding='utf-16-le', header=None)
    logger.info(f"Raw CSV shape: {df_raw.shape}")

    if df_raw.shape[0] < 5:
        st.error("CSV must have at least 5 rows: 4 headers + 1 data row")
        st.stop()

    headers = df_raw.iloc[0:4, 0].tolist()
    data_values = df_raw.iloc[4:, 0].to_numpy()

    if len(data_values) % len(headers)!= 0:
        st.error(f"Data rows {len(data_values)} not divisible by header count {len(headers)}")
        st.stop()

    num_cols = len(headers)
    num_rows = len(data_values) // num_cols
    df = pd.DataFrame(data_values.reshape(num_rows, num_cols), columns=headers)

    df.columns = df.columns.astype(str).str.lower().str.strip().str.replace(' ', '_')
    df = df.rename(columns={'is_fraud': 'Fraud', 'fraud': 'Fraud'})

    if 'Fraud' not in df.columns:
        st.error(f"Missing 'Fraud' column. Found: {list(df.columns)}")
        st.stop()

    return df

def prepare_features(df, model):
    X = df.drop('Fraud', axis=1)
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols):
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    if hasattr(model, 'feature_names_in_'):
        for col in model.feature_names_in_:
            if col not in X.columns:
                X[col] = 0
        X = X[model.feature_names_in_]
    return X

model = load_model(MODEL_PATH)

st.title("🚨 Fraud Detection Dashboard")
st.write("Upload transaction data to predict fraudulent activity")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        st.error(f"File too large. Max size: {MAX_UPLOAD_SIZE_MB} MB")
        st.stop()

    df = validate_and_process_csv(uploaded_file)
    st.success(f"File uploaded: {uploaded_file.name}")

    st.subheader("Data Preview")
    st.dataframe(df.head())

    X = prepare_features(df, model)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results_df = df.copy()
    results_df['Predicted_Fraud'] = predictions
    results_df['Fraud_Probability'] = probabilities

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(df))
    col2.metric("Predicted Fraud", int(predictions.sum()))
    col3.metric("Fraud Rate", f"{predictions.mean()*100:.2f}%")

    st.subheader("Prediction Results")
    st.dataframe(results_df)

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='fraud_predictions.csv',
        mime='text/csv'
    )

    if 'Fraud' in df.columns:
        st.subheader("Model Performance")
        y_true = df['Fraud']

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_true, predictions)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

        with col2:
            st.write("**Classification Report**")
            report = classification_report(y_true, predictions, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("👆 Upload a CSV file to start fraud detection")
    st.write("CSV format: First 4 rows should be headers, then data rows. Must include a 'Fraud' column.")
