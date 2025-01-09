# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# =============================================================================
# DATA FILTERING
# =============================================================================
def render_data_filters():
    """Render data filtering sidebar with advanced options"""
    st.sidebar.subheader("Data Filters")
    
    # Basic Filters
    filters = render_basic_filters()
    
    # Advanced Filters
    with st.sidebar.expander("Advanced Filters"):
        advanced_filters = render_advanced_filters()
    
    # Combine and apply all filters
    filtered_data = apply_filters(filters, advanced_filters)
    
    # Show filter summary
    show_filter_summary(filtered_data)
    
    return filtered_data

def render_basic_filters():
    """Render basic filtering options"""
    filters = {}
    
    # Manufacturer filter
    filters['manufacturers'] = st.sidebar.multiselect(
        "Select Manufacturers",
        options=sorted(st.session_state.data['Manufacturer'].unique()),
        default=[]
    )
    
    # Review score filters
    filters['excellent_review_range'] = st.sidebar.slider(
        "Excellent Review %",
        min_value=0,
        max_value=100,
        value=(0, 100)
    )
    
    filters['average_review_range'] = st.sidebar.slider(
        "Average Review %",
        min_value=0,
        max_value=100,
        value=(0, 100)
    )
    
    return filters

def render_advanced_filters():
    """Render advanced filtering options"""
    advanced_filters = {}
    
    # Composition complexity filter
    if 'composition_complexity' in st.session_state.data.columns:
        advanced_filters['complexity_range'] = st.slider(
            "Composition Complexity",
            min_value=float(st.session_state.data['composition_complexity'].min()),
            max_value=float(st.session_state.data['composition_complexity'].max()),
            value=(float(st.session_state.data['composition_complexity'].min()),
                  float(st.session_state.data['composition_complexity'].max()))
        )
    
    # Side effects severity filter
    if 'side_effects_severity' in st.session_state.data.columns:
        advanced_filters['severity_range'] = st.slider(
            "Side Effects Severity",
            min_value=float(st.session_state.data['side_effects_severity'].min()),
            max_value=float(st.session_state.data['side_effects_severity'].max()),
            value=(float(st.session_state.data['side_effects_severity'].min()),
                  float(st.session_state.data['side_effects_severity'].max()))
        )
    
    # Manufacturer reputation filter
    if 'manufacturer_reputation' in st.session_state.data.columns:
        advanced_filters['reputation_range'] = st.slider(
            "Manufacturer Reputation",
            min_value=float(st.session_state.data['manufacturer_reputation'].min()),
            max_value=float(st.session_state.data['manufacturer_reputation'].max()),
            value=(float(st.session_state.data['manufacturer_reputation'].min()),
                  float(st.session_state.data['manufacturer_reputation'].max()))
        )
    
    return advanced_filters

def apply_filters(filters, advanced_filters):
    """Apply all filters to the dataset"""
    data = st.session_state.data.copy()
    
    # Apply basic filters
    if filters['manufacturers']:
        data = data[data['Manufacturer'].isin(filters['manufacturers'])]
    
    data = data[
        (data['Excellent Review %'].between(
            filters['excellent_review_range'][0], 
            filters['excellent_review_range'][1]
        )) &
        (data['Average Review %'].between(
            filters['average_review_range'][0], 
            filters['average_review_range'][1]
        ))
    ]
    
    # Apply advanced filters if columns exist
    if 'complexity_range' in advanced_filters:
        data = data[data['composition_complexity'].between(
            advanced_filters['complexity_range'][0],
            advanced_filters['complexity_range'][1]
        )]
    
    if 'severity_range' in advanced_filters:
        data = data[data['side_effects_severity'].between(
            advanced_filters['severity_range'][0],
            advanced_filters['severity_range'][1]
        )]
    
    if 'reputation_range' in advanced_filters:
        data = data[data['manufacturer_reputation'].between(
            advanced_filters['reputation_range'][0],
            advanced_filters['reputation_range'][1]
        )]
    
    return data

def show_filter_summary(filtered_data):
    """Display summary of applied filters"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Summary")
    st.sidebar.write(f"Total Records: {len(filtered_data):,}")
    st.sidebar.write(f"Manufacturers: {filtered_data['Manufacturer'].nunique():,}")
    
    if 'composition_complexity' in filtered_data.columns:
        st.sidebar.write(f"Avg Complexity: {filtered_data['composition_complexity'].mean():.2f}")
    
    st.sidebar.write(f"Avg Excellent Review: {filtered_data['Excellent Review %'].mean():.1f}%")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Medicine Data Filters", page_icon="🔍", layout="wide")
    if 'data' in st.session_state:
        filtered_data = render_data_filters()
        st.write("Filtered Data Preview:", filtered_data.head())
    else:
        st.error("Please load data first!")