import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def render_data_analysis():
    """Render comprehensive data analysis page"""
    st.title("Medicine Data Analysis")

    # Sidebar controls
    st.sidebar.subheader("Analysis Controls")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Distribution Analysis", "Correlation Analysis", "Manufacturer Analysis", "Price Analysis"]
    )

    if analysis_type == "Distribution Analysis":
        render_distribution_analysis()
    elif analysis_type == "Correlation Analysis":
        render_correlation_analysis()
    elif analysis_type == "Manufacturer Analysis":
        render_manufacturer_analysis()
    elif analysis_type == "Price Analysis":
        render_price_analysis()

def render_distribution_analysis():
    """Render distribution analysis section"""
    st.subheader("Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Effectiveness Distribution
        fig_eff = px.histogram(
            st.session_state.data,
            x="Overall_Score",
            nbins=30,
            title="Distribution of Medicine Effectiveness",
            color_discrete_sequence=['rgba(0, 100, 200, 0.7)']
        )
        st.plotly_chart(fig_eff, use_container_width=True)
    
    with col2:
        # Price Distribution
        fig_price = px.histogram(
            st.session_state.data,
            x="Price",
            nbins=30,
            title="Distribution of Medicine Prices",
            color_discrete_sequence=['rgba(0, 200, 100, 0.7)']
        )
        st.plotly_chart(fig_price, use_container_width=True)

def render_correlation_analysis():
    """Render correlation analysis section"""
    st.subheader("Correlation Analysis")
    
    # Calculate correlations
    numeric_cols = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = st.session_state.data[numeric_cols].corr()
    
    # Plot correlation heatmap
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_manufacturer_analysis():
    """Render manufacturer analysis section"""
    st.subheader("Manufacturer Analysis")
    
    # Top manufacturers by average effectiveness
    top_manufacturers = (st.session_state.data.groupby('Manufacturer')
                        .agg({
                            'Overall_Score': 'mean',
                            'Price': 'mean',
                            'Medicine_Name': 'count'
                        })
                        .round(2)
                        .sort_values('Overall_Score', ascending=False)
                        .head(10))
    
    # Rename columns for display
    top_manufacturers.columns = ['Avg Effectiveness', 'Avg Price', 'Number of Medicines']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_mfg = px.bar(
            top_manufacturers,
            y=top_manufacturers.index,
            x='Avg Effectiveness',
            title="Top Manufacturers by Effectiveness",
            orientation='h'
        )
        st.plotly_chart(fig_mfg, use_container_width=True)
    
    with col2:
        st.dataframe(top_manufacturers)

def render_price_analysis():
    """Render price analysis section"""
    st.subheader("Price Analysis")
    
    # Price range selector
    price_range = st.slider(
        "Select Price Range",
        min_value=float(st.session_state.data['Price'].min()),
        max_value=float(st.session_state.data['Price'].max()),
        value=(float(st.session_state.data['Price'].min()),
               float(st.session_state.data['Price'].max()))
    )
    
    # Filter data based on price range
    filtered_data = st.session_state.data[
        st.session_state.data['Price'].between(price_range[0], price_range[1])
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price vs Effectiveness scatter plot
        fig_scatter = px.scatter(
            filtered_data,
            x="Price",
            y="Overall_Score",
            color="Manufacturer",
            title="Price vs Effectiveness",
            trendline="ols"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Price statistics
        st.subheader("Price Statistics")
        price_stats = filtered_data['Price'].describe().round(2)
        st.dataframe(pd.DataFrame(price_stats))