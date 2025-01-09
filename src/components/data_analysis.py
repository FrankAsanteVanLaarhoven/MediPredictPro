# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# MAIN ANALYSIS RENDERER
# =============================================================================
def render_data_analysis():
    """Render comprehensive data analysis page"""
    st.title("Medicine Data Analysis")
    
    analysis_type = render_sidebar_controls()
    
    analysis_functions = {
        "Distribution Analysis": render_distribution_analysis,
        "Correlation Analysis": render_correlation_analysis,
        "Manufacturer Analysis": render_manufacturer_analysis,
        "Review Analysis": render_review_analysis,
        "Composition Analysis": render_composition_analysis
    }
    
    analysis_functions[analysis_type]()

def render_sidebar_controls():
    """Render sidebar controls for analysis"""
    st.sidebar.subheader("Analysis Controls")
    return st.sidebar.selectbox(
        "Select Analysis Type",
        ["Distribution Analysis", "Correlation Analysis", 
         "Manufacturer Analysis", "Review Analysis", 
         "Composition Analysis"]
    )

# =============================================================================
# DISTRIBUTION ANALYSIS
# =============================================================================
def render_distribution_analysis():
    """Render distribution analysis section"""
    st.subheader("Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_review_distribution()
    with col2:
        render_manufacturer_distribution()
    
    render_review_statistics()

def render_review_distribution():
    """Render review score distributions"""
    fig = px.histogram(
        st.session_state.data,
        x="Excellent Review %",
        nbins=30,
        title="Distribution of Excellent Reviews",
        color_discrete_sequence=['rgba(0, 100, 200, 0.7)']
    )
    st.plotly_chart(fig, use_container_width=True)

def render_manufacturer_distribution():
    """Render manufacturer distribution"""
    top_manufacturers = (st.session_state.data['Manufacturer']
                        .value_counts()
                        .head(10))
    
    fig = px.bar(
        x=top_manufacturers.values,
        y=top_manufacturers.index,
        orientation='h',
        title="Top 10 Manufacturers by Number of Medicines"
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================
def render_correlation_analysis():
    """Render correlation analysis section"""
    st.subheader("Correlation Analysis")
    
    review_cols = ['Excellent Review %', 'Average Review %', 'Poor Review %']
    corr_matrix = st.session_state.data[review_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Review Score Correlations",
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    render_review_relationships()

def render_review_relationships():
    """Render relationships between review scores"""
    fig = px.scatter_matrix(
        st.session_state.data,
        dimensions=['Excellent Review %', 'Average Review %', 'Poor Review %'],
        title="Review Score Relationships"
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MANUFACTURER ANALYSIS
# =============================================================================
def render_manufacturer_analysis():
    """Render manufacturer analysis section"""
    st.subheader("Manufacturer Analysis")
    
    manufacturer_metrics = calculate_manufacturer_metrics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_manufacturer_performance(manufacturer_metrics)
    with col2:
        render_manufacturer_details(manufacturer_metrics)

def calculate_manufacturer_metrics():
    """Calculate manufacturer performance metrics"""
    return (st.session_state.data.groupby('Manufacturer')
            .agg({
                'Excellent Review %': 'mean',
                'Average Review %': 'mean',
                'Poor Review %': 'mean',
                'Medicine Name': 'count'
            })
            .round(2)
            .sort_values('Excellent Review %', ascending=False))

# =============================================================================
# REVIEW ANALYSIS
# =============================================================================
def render_review_analysis():
    """Render review analysis section"""
    st.subheader("Review Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_review_trends()
    with col2:
        render_review_breakdown()

def render_review_trends():
    """Render review score trends"""
    review_means = pd.DataFrame({
        'Type': ['Excellent', 'Average', 'Poor'],
        'Percentage': [
            st.session_state.data['Excellent Review %'].mean(),
            st.session_state.data['Average Review %'].mean(),
            st.session_state.data['Poor Review %'].mean()
        ]
    })
    
    fig = px.bar(
        review_means,
        x='Type',
        y='Percentage',
        title="Average Review Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# COMPOSITION ANALYSIS
# =============================================================================
def render_composition_analysis():
    """Render composition analysis section"""
    st.subheader("Composition Analysis")
    
    composition_stats = analyze_compositions()
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_composition_complexity(composition_stats)
    with col2:
        render_composition_details(composition_stats)

def analyze_compositions():
    """Analyze medicine compositions"""
    return (st.session_state.data['Composition']
            .str.split('+')
            .agg(['count', 'unique']))

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def render_statistics_table(data: pd.DataFrame, title: str):
    """Render a formatted statistics table"""
    st.subheader(title)
    st.dataframe(
        data.style.background_gradient(cmap='Blues'),
        use_container_width=True
    )

def create_plotly_figure(data, plot_type: str, **kwargs):
    """Create a plotly figure with consistent styling"""
    plot_functions = {
        'bar': px.bar,
        'scatter': px.scatter,
        'histogram': px.histogram,
        'box': px.box
    }
    
    fig = plot_functions[plot_type](data, **kwargs)
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        margin=dict(t=50, l=0, r=0, b=0)
    )
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Medicine Data Analysis", page_icon="📊", layout="wide")
    if 'data' in st.session_state:
        render_data_analysis()
    else:
        st.error("Please load data first!")