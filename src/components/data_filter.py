import streamlit as st
import pandas as pd

def render_data_filters():
    """Render data filtering sidebar"""
    st.sidebar.subheader("Data Filters")
    
    # Manufacturer filter
    manufacturers = st.sidebar.multiselect(
        "Select Manufacturers",
        options=sorted(st.session_state.data['Manufacturer'].unique()),
        default=[]
    )
    
    # Price range filter
    price_range = st.sidebar.slider(
        "Price Range",
        min_value=float(st.session_state.data['Price'].min()),
        max_value=float(st.session_state.data['Price'].max()),
        value=(float(st.session_state.data['Price'].min()),
               float(st.session_state.data['Price'].max()))
    )
    
    # Effectiveness range filter
    effectiveness_range = st.sidebar.slider(
        "Effectiveness Range",
        min_value=0,
        max_value=100,
        value=(0, 100)
    )
    
    # Apply filters
    filtered_data = filter_data(manufacturers, price_range, effectiveness_range)
    
    return filtered_data

def filter_data(manufacturers, price_range, effectiveness_range):
    """Apply filters to the dataset"""
    data = st.session_state.data.copy()
    
    # Apply manufacturer filter
    if manufacturers:
        data = data[data['Manufacturer'].isin(manufacturers)]
    
    # Apply price filter
    data = data[data['Price'].between(price_range[0], price_range[1])]
    
    # Apply effectiveness filter
    data = data[data['Overall_Score'].between(effectiveness_range[0], effectiveness_range[1])]
    
    return data