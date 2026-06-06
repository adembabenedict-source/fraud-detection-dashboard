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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Config
MODEL_PATH = os.getenv("MODEL_PATH", "fraud_model.pkl")
MAX_UPLOAD_SIZE_MB = 200

@st.cache_resource
def load_model(path):
    """Load model once and cache it."""
    if not Path(path).exists():
        st.error(f"Model file not found at {path}")
        st.stop()
    logger.info(f"Loading model from {path}")
    return joblib.load(path)

def validate_and_process_csv(uploaded_file, model_features=None):
    """Validate CSV structure and process into model-ready format."""
    try:
        df_raw = pd.read_csv(uploaded_file, encoding='utf-16-le', header=None)
        logger.info(f"Raw CSV shape: {df_raw.shape}")

        if df_raw.shape[0] < 5:
            raise ValueError("CSV must have at least 5 rows: 4 headers + 1 data row")

        headers = df_raw.iloc[0:4, 0].tolist()
        data_values = df_raw.iloc[4:, 0].to_numpy()

        if len(data_values) % len(headers)!= 0:
            raise ValueError(f"Data rows {len(data_values)} not divisible by header count {len(headers)}")

        num_cols = len(headers)
        num_rows = len(data_values) // num_cols
        df = pd.DataFrame(data_values.reshape(num_rows, num_cols), columns=headers)

        df.columns = df.columns.astype(str).str.lower().str.strip().str.replace(' ', '_')
        df = df.rename(columns={'is_fraud': 'Fraud', 'fraud': 'Fraud'})

        if 'Fraud' not in df.columns:
            raise ValueError(f"Missing 'Fraud' column. Found: {list(df.columns)}")

        return df
    except Exception as e:
        logger.error(f"CSV processing failed: {e}")
        raise

def prepare_features(df, model):
    """One-hot encode and align features to model training."""
    X = df.drop('Fraud', axis=1)

    # Handle categoricals
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols):
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    # Force numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Align to training features
    if hasattr(model, 'feature_names_in_'):
        for col in model.feature_names_in_:
            if col not in X.columns:
                X[col] = 0
        X = X[model.feature_names_in_]

    return X

# Load model
model = load_model(MODEL_PATH)

st.title("🚨 Fraud Detection Dashboard")
st.write("Upload transaction data to predict fraudulent activity")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        st.error(f"File too large. Max size: {MAX_UPLOAD_SIZE_MB} MB")
        st.stop()

    try:
        df = validate_and_process_csv(uploaded_file)
        st.success(f"File uploaded: {uploaded_file.name}")

        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Prepare features and predict
        X = prepare_features(df, model)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        results_df = df.copy()
        results_df['Predicted_Fraud'] = predictions
        results_df['Fraud_Probability'] = probabilities

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", len(df))
        col2.metric("Predicted Fraud", int(predictions.sum()))
        col3.metric("Fraud Rate", f"{predictions.mean()*100:.2f}%")

        # Show results
        st.subheader("Prediction Results")
        st.dataframe(results_df)

        # Download results
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='fraud_predictions.csv',
            mime='text/csv'
        )

        # If actual labels exist, show metrics
        if 'Fraud' in df.columns:
            st.sub
