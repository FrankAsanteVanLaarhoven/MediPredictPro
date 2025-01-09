# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from datetime import datetime, timedelta

# =============================================================================
# SIDEBAR RENDERING
# =============================================================================
def render_sidebar():
    """Render sidebar with navigation and data controls"""
    with st.sidebar:
        selected = render_navigation()
        render_data_controls()
        render_filters()
        render_settings()
        render_footer()
        
    return selected

def render_navigation():
    """Render navigation menu"""
    return option_menu(
        menu_title="Navigation",
        options=["Dashboard", "Analysis", "Predictions"],
        icons=["house", "graph-up", "cpu"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "#00A67E", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#00A67E"},
        }
    )

# =============================================================================
# DATA CONTROLS
# =============================================================================
def render_data_controls():
    """Render data loading and control section"""
    st.header("Data Controls")
    
    initialize_session_state()
    
    if st.button("Load Data", key="load_data_btn"):
        load_data_with_feedback()

def initialize_session_state():
    """Initialize session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'filters' not in st.session_state:
        st.session_state.filters = {}

def load_data_with_feedback():
    """Load data with user feedback"""
    with st.spinner("Loading data..."):
        from .data_loader import load_data
        st.session_state.data = load_data()
        if st.session_state.data is not None:
            st.success(f"Data loaded: {len(st.session_state.data):,} records")
        else:
            st.error("Failed to load data")

# =============================================================================
# FILTERS
# =============================================================================
def render_filters():
    """Render filter controls"""
    if st.session_state.data is not None:
        st.subheader("Filters")
        
        render_manufacturer_filter()
        render_review_filters()
        render_date_filter()
        render_filter_controls()
        show_active_filters()

def render_manufacturer_filter():
    """Render manufacturer selection filter"""
    manufacturers = sorted(st.session_state.data['Manufacturer'].unique())
    st.session_state.filters['manufacturers'] = st.multiselect(
        "Select Manufacturers",
        options=manufacturers,
        default=manufacturers[:5],
        key="manufacturer_filter"
    )

def render_review_filters():
    """Render review-related filters"""
    # Excellent Review Filter
    excellent_range = st.slider(
        "Excellent Review %",
        min_value=0,
        max_value=100,
        value=(0, 100),
        key="excellent_review_filter"
    )
    st.session_state.filters['excellent_range'] = excellent_range
    
    # Average Review Filter
    average_range = st.slider(
        "Average Review %",
        min_value=0,
        max_value=100,
        value=(0, 100),
        key="average_review_filter"
    )
    st.session_state.filters['average_range'] = average_range

def render_date_filter():
    """Render date range filter if applicable"""
    if 'Date' in st.session_state.data.columns:
        min_date = st.session_state.data['Date'].min()
        max_date = st.session_state.data['Date'].max()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_filter"
        )
        st.session_state.filters['date_range'] = date_range

def render_filter_controls():
    """Render filter control buttons"""
    if st.button("Reset Filters", key="reset_filters_btn"):
        st.session_state.filters = {}
        st.experimental_rerun()

def show_active_filters():
    """Display active filters"""
    with st.expander("Active Filters"):
        for filter_name, filter_value in st.session_state.filters.items():
            st.write(f"{filter_name}: {filter_value}")

# =============================================================================
# SETTINGS AND FOOTER
# =============================================================================
def render_settings():
    """Render settings section"""
    with st.expander("Settings"):
        st.checkbox("Dark Mode", key="dark_mode")
        st.selectbox(
            "Update Frequency",
            ["Real-time", "Daily", "Weekly"],
            key="update_freq"
        )

def render_footer():
    """Render sidebar footer"""
    st.divider()
    st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# =============================================================================
# FILTER APPLICATION
# =============================================================================
def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    """Apply active filters to the dataset"""
    if data is None or not st.session_state.filters:
        return data
    
    filtered_data = data.copy()
    
    # Apply manufacturer filter
    if should_apply_manufacturer_filter():
        filtered_data = apply_manufacturer_filter(filtered_data)
    
    # Apply review filters
    filtered_data = apply_review_filters(filtered_data)
    
    # Apply date filter
    if should_apply_date_filter():
        filtered_data = apply_date_filter(filtered_data)
    
    return filtered_data

def should_apply_manufacturer_filter():
    """Check if manufacturer filter should be applied"""
    return ('manufacturers' in st.session_state.filters and 
            st.session_state.filters['manufacturers'])

def apply_manufacturer_filter(data):
    """Apply manufacturer filter to data"""
    return data[data['Manufacturer'].isin(st.session_state.filters['manufacturers'])]

def apply_review_filters(data):
    """Apply review-related filters to data"""
    if 'excellent_range' in st.session_state.filters:
        min_exc, max_exc = st.session_state.filters['excellent_range']
        data = data[
            (data['Excellent Review %'] >= min_exc) & 
            (data['Excellent Review %'] <= max_exc)
        ]
    
    if 'average_range' in st.session_state.filters:
        min_avg, max_avg = st.session_state.filters['average_range']
        data = data[
            (data['Average Review %'] >= min_avg) & 
            (data['Average Review %'] <= max_avg)
        ]
    
    return data

def should_apply_date_filter():
    """Check if date filter should be applied"""
    return ('date_range' in st.session_state.filters and 
            'Date' in st.session_state.data.columns)

def apply_date_filter(data):
    """Apply date filter to data"""
    start_date, end_date = st.session_state.filters['date_range']
    return data[
        (data['Date'] >= start_date) & 
        (data['Date'] <= end_date)
    ]

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Medicine Analysis", page_icon="💊", layout="wide")
    selected = render_sidebar()