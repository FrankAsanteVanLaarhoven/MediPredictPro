import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def render_visualization_dashboard():
    """Render the data visualization dashboard"""
    st.title("Medicine Data Visualization Dashboard")
    
    # Create tabs for different visualization categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribution Analysis",
        "Correlation Analysis",
        "Manufacturer Insights",
        "Custom Analysis"
    ])
    
    with tab1:
        render_distribution_analysis()
    
    with tab2:
        render_correlation_analysis()
    
    with tab3:
        render_manufacturer_insights()
    
    with tab4:
        render_custom_analysis()

def render_distribution_analysis():
    """Render distribution analysis section"""
    st.subheader("Distribution Analysis")
    
    # Select variable for analysis
    variable = st.selectbox(
        "Select Variable",
        options=['Overall_Score', 'Price', 'Composition_Length', 'Side_Effects_Count'],
        help="Choose variable to analyze"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig_hist = px.histogram(
            st.session_state.data,
            x=variable,
            nbins=30,
            title=f"Distribution of {variable}",
            color_discrete_sequence=['rgba(0, 100, 200, 0.7)']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot
        fig_box = px.box(
            st.session_state.data,
            y=variable,
            title=f"Box Plot of {variable}"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    stats = st.session_state.data[variable].describe()
    st.dataframe(pd.DataFrame(stats).T)

def render_correlation_analysis():
    """Render correlation analysis section"""
    st.subheader("Correlation Analysis")
    
    # Select variables for correlation
    numeric_cols = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns
    variables = st.multiselect(
        "Select Variables",
        options=numeric_cols,
        default=list(numeric_cols)[:4],
        help="Choose variables for correlation analysis"
    )
    
    if variables:
        # Correlation matrix
        corr_matrix = st.session_state.data[variables].corr()
        
        # Heatmap
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot matrix
        if len(variables) > 1:
            fig_scatter = px.scatter_matrix(
                st.session_state.data[variables],
                dimensions=variables,
                title="Scatter Plot Matrix"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

def render_manufacturer_insights():
    """Render manufacturer analysis section"""
    st.subheader("Manufacturer Insights")
    
    # Top manufacturers analysis
    top_n = st.slider("Select Top N Manufacturers", 5, 20, 10)
    
    # Aggregate metrics by manufacturer
    manufacturer_metrics = st.session_state.data.groupby('Manufacturer').agg({
        'Overall_Score': ['mean', 'std', 'count'],
        'Price': 'mean'
    }).round(2)
    
    manufacturer_metrics.columns = ['Avg Score', 'Std Score', 'Medicine Count', 'Avg Price']
    top_manufacturers = manufacturer_metrics.nlargest(top_n, 'Avg Score')
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(
            top_manufacturers,
            y=top_manufacturers.index,
            x='Avg Score',
            error_x='Std Score',
            title=f"Top {top_n} Manufacturers by Average Score",
            orientation='h'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        fig_scatter = px.scatter(
            top_manufacturers,
            x='Avg Price',
            y='Avg Score',
            size='Medicine Count',
            hover_data=['Std Score'],
            text=top_manufacturers.index,
            title="Price vs Score by Manufacturer"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Detailed metrics table
    st.dataframe(top_manufacturers.style.background_gradient(subset=['Avg Score']))

def render_custom_analysis():
    """Render custom analysis section"""
    st.subheader("Custom Analysis")
    
    # Select analysis type
    analysis_type = st.selectbox(
        "Select Analysis Type",
        options=['Scatter Plot', 'Bar Chart', 'Line Chart', 'Box Plot'],
        help="Choose type of analysis"
    )
    
    # Select variables
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Select X Variable", options=st.session_state.data.columns)
        color_var = st.selectbox("Select Color Variable (optional)", 
                               options=['None'] + list(st.session_state.data.columns))
    
    with col2:
        y_var = st.selectbox("Select Y Variable", options=st.session_state.data.columns)
        size_var = st.selectbox("Select Size Variable (optional)",
                              options=['None'] + list(st.session_state.data.columns))
    
    # Create visualization
    if analysis_type == 'Scatter Plot':
        fig = create_scatter_plot(x_var, y_var, color_var, size_var)
    elif analysis_type == 'Bar Chart':
        fig = create_bar_chart(x_var, y_var, color_var)
    elif analysis_type == 'Line Chart':
        fig = create_line_chart(x_var, y_var, color_var)
    else:
        fig = create_box_plot(x_var, y_var, color_var)
    
    st.plotly_chart(fig, use_container_width=True)

def create_scatter_plot(x_var, y_var, color_var, size_var):
    """Create custom scatter plot"""
    plot_kwargs = {
        'data_frame': st.session_state.data,
        'x': x_var,
        'y': y_var,
        'title': f'{y_var} vs {x_var}'
    }
    
    if color_var != 'None':
        plot_kwargs['color'] = color_var
    
    if size_var != 'None':
        plot_kwargs['size'] = size_var
    
    return px.scatter(**plot_kwargs)

# Add similar functions for other plot types
def create_bar_chart(x_var, y_var, color_var):
    """Create custom bar chart"""
    plot_kwargs = {
        'data_frame': st.session_state.data,
        'x': x_var,
        'y': y_var,
        'title': f'{y_var} by {x_var}'
    }
    
    if color_var != 'None':
        plot_kwargs['color'] = color_var
    
    return px.bar(**plot_kwargs)

def create_line_chart(x_var, y_var, color_var):
    """Create custom line chart"""
    plot_kwargs = {
        'data_frame': st.session_state.data,
        'x': x_var,
        'y': y_var,
        'title': f'{y_var} over {x_var}'
    }
    
    if color_var != 'None':
        plot_kwargs['color'] = color_var
    
    return px.line(**plot_kwargs)

def create_box_plot(x_var, y_var, color_var):
    """Create custom box plot"""
    plot_kwargs = {
        'data_frame': st.session_state.data,
        'x': x_var,
        'y': y_var,
        'title': f'Distribution of {y_var} by {x_var}'
    }
    
    if color_var != 'None':
        plot_kwargs['color'] = color_var
    
    return px.box(**plot_kwargs)