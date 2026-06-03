import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Config from environment variables
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

def validate_and_process_csv(uploaded_file, model_features):
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
        for col
