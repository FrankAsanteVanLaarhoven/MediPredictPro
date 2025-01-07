import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import time
from datetime import datetime, timedelta

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
    st.subheader("Price Analysis")
    
    fig = px.histogram(data, x='Price', nbins=50, title='Price Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.box(data, x='Manufacturer', y='Price', title='Price by Manufacturer')
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

def render_effectiveness_analysis(data: pd.DataFrame):
    st.subheader("Effectiveness Analysis")
    
    if 'Overall_Score' in data.columns:
        fig = px.histogram(data, x='Overall_Score', nbins=20)
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.scatter(data, x='Price', y='Overall_Score', color='Manufacturer')
        st.plotly_chart(fig, use_container_width=True)

def render_manufacturer_comparison(data: pd.DataFrame):
    st.subheader("Manufacturer Comparison")
    
    fig = px.pie(data, names='Manufacturer', title='Market Share')
    st.plotly_chart(fig, use_container_width=True)
    
    stats = data.groupby('Manufacturer').agg({
        'Price': 'mean',
        'Overall_Score': 'mean' if 'Overall_Score' in data.columns else 'count'
    }).reset_index()
    
    fig = px.bar(stats, x='Manufacturer', y=['Price', 'Overall_Score'])
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=3600)
def process_data_for_analysis(data: pd.DataFrame) -> pd.DataFrame:
    return data.copy()
