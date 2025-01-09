# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import time
from datetime import datetime, timedelta

# =============================================================================
# ANALYSIS PAGE FUNCTIONS
# =============================================================================
def render_analysis(data: pd.DataFrame):
    """Render the analysis page with various analytical components"""
    st.title("Medicine Analysis 📊")
    
    if data is None or data.empty:
        st.warning("No data available for analysis")
        return

    with st.spinner("Analyzing data..."):
        time.sleep(0.5)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview Statistics",
            "Price Analysis", 
            "Effectiveness Analysis",
            "Manufacturer Comparison"
        ])
        
        with tab1:
            render_overview_statistics(data)
        with tab2:
            render_price_analysis(data)
        with tab3:
            render_effectiveness_analysis(data)
        with tab4:
            render_manufacturer_comparison(data)

def render_overview_statistics(data: pd.DataFrame):
    """Render overview statistics section"""
    st.subheader("Overview Statistics")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Medicines", f"{len(data):,}")
    with col2:
        avg_price = data['Price'].mean()
        st.metric("Average Price", f"${avg_price:,.2f}")
    with col3:
        if 'Overall_Score' in data.columns:
            avg_score = data['Overall_Score'].mean()
            st.metric("Average Effectiveness", f"{avg_score:.1f}")

def render_price_analysis(data: pd.DataFrame):
    """Render price analysis section"""
    st.subheader("Price Analysis")
    
    # Price distribution histogram
    fig = px.histogram(
        data, 
        x='Price', 
        nbins=50, 
        title='Price Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Price by manufacturer box plot
    fig = px.box(
        data, 
        x='Manufacturer', 
        y='Price', 
        title='Price by Manufacturer'
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

def render_effectiveness_analysis(data: pd.DataFrame):
    """Render effectiveness analysis section"""
    st.subheader("Effectiveness Analysis")
    
    if 'Overall_Score' in data.columns:
        # Effectiveness distribution
        fig = px.histogram(
            data, 
            x='Overall_Score', 
            nbins=20,
            title='Effectiveness Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Price vs Effectiveness scatter plot
        fig = px.scatter(
            data, 
            x='Price', 
            y='Overall_Score', 
            color='Manufacturer',
            title='Price vs Effectiveness by Manufacturer'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Effectiveness data not available")

def render_manufacturer_comparison(data: pd.DataFrame):
    """Render manufacturer comparison section"""
    st.subheader("Manufacturer Comparison")
    
    # Market share pie chart
    fig = px.pie(
        data, 
        names='Manufacturer', 
        title='Market Share by Manufacturer'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Manufacturer statistics
    stats = data.groupby('Manufacturer').agg({
        'Price': 'mean',
        'Overall_Score': 'mean' if 'Overall_Score' in data.columns else 'count'
    }).reset_index()
    
    # Comparative bar chart
    fig = px.bar(
        stats, 
        x='Manufacturer', 
        y=['Price', 'Overall_Score'],
        title='Price and Effectiveness by Manufacturer'
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# DATA PROCESSING
# =============================================================================
@st.cache_data(ttl=3600)
def process_data_for_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process and cache data for analysis
    
    Args:
        data (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    try:
        processed_data = data.copy()
        
        # Add any data processing steps here
        # For example:
        # - Handle missing values
        # - Convert data types
        # - Calculate additional metrics
        
        return processed_data
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return data

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # This section will only run if the file is run directly
    st.set_page_config(
        page_title="Medicine Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    # Load sample data for testing
    sample_data = pd.DataFrame({
        'Medicine_Name': [f'Med_{i}' for i in range(100)],
        'Price': np.random.uniform(10, 1000, 100),
        'Manufacturer': np.random.choice(['Pfizer', 'Novartis', 'Roche'], 100),
        'Overall_Score': np.random.uniform(60, 100, 100)
    })
    
    # Render analysis page
    render_analysis(sample_data)