import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Fraud Detection Dashboard", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for black/blue/white theme
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    .metric-card { background-color: #1E3A8A; padding: 20px; border-radius: 10px; }
    h1, h2, h3 { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

st.title("Fraud Detection Dashboard")
st.write("Real-time transaction monitoring with ML-powered fraud detection")

# Generate demo data - this runs automatically
@st.cache_data
def load_demo_data():
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=500, freq='H')
    merchants = ['Amazon', 'Walmart', 'Target', 'Unknown_Shop', 'Gas_Station', 'Grocery_Store']
    
    df = pd.DataFrame({
        'date': dates,
        'amount': np.random.exponential(100, 500) + 10,
        'merchant': np.random.choice(merchants, 500),
        'location': np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL'], 500),
        'is_fraud': np.random.choice([0, 1], 500, p=[0.95, 0.05])
    })
    # Make fraud amounts higher
    df.loc[df['is_fraud'] == 1, 'amount'] = df.loc[df['is_fraud'] == 1, 'amount'] * 5
    return df

# Load data
df = load_demo_data()

# Option to upload real
