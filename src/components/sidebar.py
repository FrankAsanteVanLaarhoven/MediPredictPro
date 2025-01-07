# src/components/sidebar.py

import streamlit as st
from streamlit_option_menu import option_menu
from .data_loader import load_data
import pandas as pd
from datetime import datetime, timedelta

def render_sidebar():
    """Render sidebar with navigation and data controls"""
    with st.sidebar:
        # Navigation Menu
        selected = option_menu(
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
        
        # Data Controls Section
        st.header("Data Controls")
        
        # Initialize session state if needed
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'filters' not in st.session_state:
            st.session_state.filters = {}
        
        # Load Data Button
        if st.button("Load Data", key="load_data_btn"):
            with st.spinner("Loading data..."):
                st.session_state.data = load_data()
                if st.session_state.data is not None:
                    st.success(f"Data loaded: {len(st.session_state.data):,} records")
                else:
                    st.error("Failed to load data")
        
        # Filters Section
        if st.session_state.data is not None:
            st.subheader("Filters")
            
            # Manufacturer Filter
            manufacturers = sorted(st.session_state.data['Manufacturer'].unique())
            st.session_state.filters['manufacturers'] = st.multiselect(
                "Select Manufacturers",
                options=manufacturers,
                default=manufacturers[:5],
                key="manufacturer_filter"
            )
            
            # Price Range Filter
            price_range = st.slider(
                "Price Range",
                min_value=float(st.session_state.data['Price'].min()),
                max_value=float(st.session_state.data['Price'].max()),
                value=(float(st.session_state.data['Price'].min()),
                       float(st.session_state.data['Price'].max())),
                key="price_filter"
            )
            st.session_state.filters['price_range'] = price_range
            
            # Effectiveness Score Filter
            if 'Overall_Score' in st.session_state.data.columns:
                score_range = st.slider(
                    "Effectiveness Score",
                    min_value=0,
                    max_value=100,
                    value=(0, 100),
                    key="score_filter"
                )
                st.session_state.filters['score_range'] = score_range
            
            # Date Range Filter (if applicable)
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
            
            # Reset Filters Button
            if st.button("Reset Filters", key="reset_filters_btn"):
                st.session_state.filters = {}
                st.experimental_rerun()
            
            # Show Active Filters
            with st.expander("Active Filters"):
                for filter_name, filter_value in st.session_state.filters.items():
                    st.write(f"{filter_name}: {filter_value}")
        
        # Additional Controls
        with st.expander("Settings"):
            st.checkbox("Dark Mode", key="dark_mode")
            st.selectbox(
                "Update Frequency",
                ["Real-time", "Daily", "Weekly"],
                key="update_freq"
            )
        
        # Footer
        st.divider()
        st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
    return selected

def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    """Apply active filters to the dataset"""
    if data is None or not st.session_state.filters:
        return data
    
    filtered_data = data.copy()
    
    # Apply manufacturer filter
    if 'manufacturers' in st.session_state.filters and st.session_state.filters['manufacturers']:
        filtered_data = filtered_data[
            filtered_data['Manufacturer'].isin(st.session_state.filters['manufacturers'])
        ]
    
    # Apply price range filter
    if 'price_range' in st.session_state.filters:
        min_price, max_price = st.session_state.filters['price_range']
        filtered_data = filtered_data[
            (filtered_data['Price'] >= min_price) & 
            (filtered_data['Price'] <= max_price)
        ]
    
    # Apply effectiveness score filter
    if 'score_range' in st.session_state.filters:
        min_score, max_score = st.session_state.filters['score_range']
        filtered_data = filtered_data[
            (filtered_data['Overall_Score'] >= min_score) & 
            (filtered_data['Overall_Score'] <= max_score)
        ]
    
    # Apply date range filter
    if 'date_range' in st.session_state.filters:
        start_date, end_date = st.session_state.filters['date_range']
        filtered_data = filtered_data[
            (filtered_data['Date'] >= start_date) & 
            (filtered_data['Date'] <= end_date)
        ]
    
    return filtered_data