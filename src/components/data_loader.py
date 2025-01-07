import streamlit as st
import pandas as pd
from pathlib import Path

def initialize_session_state():
    """Initialize session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []

def load_data():
    """Load medicine dataset"""
    try:
        data_path = Path('data/Medicine_Details.csv')
        if data_path.exists():
            df = pd.read_csv(data_path)
            st.session_state.data = df
            return df
        else:
            st.error("Data file not found in data directory")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None