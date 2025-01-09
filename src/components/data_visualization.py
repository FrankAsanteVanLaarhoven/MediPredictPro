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
# MAIN VISUALIZATION DASHBOARD
# =============================================================================
def render_visualization_dashboard():
    """Render the data visualization dashboard"""
    st.title("Medicine Data Visualization Dashboard")
    
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

# =============================================================================
# DISTRIBUTION ANALYSIS
# =============================================================================
def render_distribution_analysis():
    """Render distribution analysis section"""
    st.subheader("Distribution Analysis")
    
    variable = st.selectbox(
        "Select Variable",
        options=['Overall_Score', 'Price', 'Composition_Length', 'Side_Effects_Count'],
        help="Choose variable to analyze"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_histogram(variable)
    
    with col2:
        render_boxplot(variable)
    
    render_summary_statistics(variable)

def render_histogram(variable):
    """Render histogram for selected variable"""
    fig_hist = px.histogram(
        st.session_state.data,
        x=variable,
        nbins=30,
        title=f"Distribution of {variable}",
        color_discrete_sequence=['rgba(0, 100, 200, 0.7)']
    )
    st.plotly_chart(fig_hist, use_container_width=True)

def render_boxplot(variable):
    """Render box plot for selected variable"""
    fig_box = px.box(
        st.session_state.data,
        y=variable,
        title=f"Box Plot of {variable}"
    )
    st.plotly_chart(fig_box, use_container_width=True)

def render_summary_statistics(variable):
    """Render summary statistics for selected variable"""
    st.subheader("Summary Statistics")
    stats = st.session_state.data[variable].describe()
    st.dataframe(pd.DataFrame(stats).T)

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================
def render_correlation_analysis():
    """Render correlation analysis section"""
    st.subheader("Correlation Analysis")
    
    numeric_cols = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns
    variables = st.multiselect(
        "Select Variables",
        options=numeric_cols,
        default=list(numeric_cols)[:4],
        help="Choose variables for correlation analysis"
    )
    
    if variables:
        render_correlation_matrix(variables)
        if len(variables) > 1:
            render_scatter_matrix(variables)

def render_correlation_matrix(variables):
    """Render correlation matrix heatmap"""
    corr_matrix = st.session_state.data[variables].corr()
    fig = px.imshow(
        corr_matrix,
        title="Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_scatter_matrix(variables):
    """Render scatter plot matrix"""
    fig_scatter = px.scatter_matrix(
        st.session_state.data[variables],
        dimensions=variables,
        title="Scatter Plot Matrix"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# =============================================================================
# MANUFACTURER INSIGHTS
# =============================================================================
def render_manufacturer_insights():
    """Render manufacturer analysis section"""
    st.subheader("Manufacturer Insights")
    
    top_n = st.slider("Select Top N Manufacturers", 5, 20, 10)
    manufacturer_metrics = calculate_manufacturer_metrics()
    top_manufacturers = manufacturer_metrics.nlargest(top_n, 'Avg Score')
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_manufacturer_bar_chart(top_manufacturers)
    
    with col2:
        render_manufacturer_scatter_plot(top_manufacturers)
    
    render_manufacturer_metrics_table(top_manufacturers)

def calculate_manufacturer_metrics():
    """Calculate aggregated metrics by manufacturer"""
    return st.session_state.data.groupby('Manufacturer').agg({
        'Overall_Score': ['mean', 'std', 'count'],
        'Price': 'mean'
    }).round(2).rename(columns={
        'Overall_Score': {'mean': 'Avg Score', 'std': 'Std Score', 'count': 'Medicine Count'},
        'Price': {'mean': 'Avg Price'}
    })

def render_manufacturer_bar_chart(top_manufacturers):
    """Render bar chart for top manufacturers"""
    fig_bar = px.bar(
        top_manufacturers,
        y=top_manufacturers.index,
        x='Avg Score',
        error_x='Std Score',
        title=f"Top {len(top_manufacturers)} Manufacturers by Average Score",
        orientation='h'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

def render_manufacturer_scatter_plot(top_manufacturers):
    """Render scatter plot for manufacturer metrics"""
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

def render_manufacturer_metrics_table(top_manufacturers):
    """Render detailed metrics table"""
    st.dataframe(top_manufacturers.style.background_gradient(subset=['Avg Score']))

# =============================================================================
# CUSTOM ANALYSIS
# =============================================================================
def render_custom_analysis():
    """Render custom analysis section"""
    st.subheader("Custom Analysis")
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        options=['Scatter Plot', 'Bar Chart', 'Line Chart', 'Box Plot'],
        help="Choose type of analysis"
    )
    
    variables = select_plot_variables()
    fig = create_custom_plot(analysis_type, **variables)
    st.plotly_chart(fig, use_container_width=True)

def select_plot_variables():
    """Select variables for custom plot"""
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Select X Variable", options=st.session_state.data.columns)
        color_var = st.selectbox("Select Color Variable (optional)", 
                               options=['None'] + list(st.session_state.data.columns))
    
    with col2:
        y_var = st.selectbox("Select Y Variable", options=st.session_state.data.columns)
        size_var = st.selectbox("Select Size Variable (optional)",
                              options=['None'] + list(st.session_state.data.columns))
    
    return {
        'x_var': x_var,
        'y_var': y_var,
        'color_var': color_var,
        'size_var': size_var
    }

def create_custom_plot(analysis_type, x_var, y_var, color_var, size_var):
    """Create custom plot based on selected type and variables"""
    plot_functions = {
        'Scatter Plot': create_scatter_plot,
        'Bar Chart': create_bar_chart,
        'Line Chart': create_line_chart,
        'Box Plot': create_box_plot
    }
    
    return plot_functions[analysis_type](x_var, y_var, color_var, size_var)

# =============================================================================
# PLOT CREATION FUNCTIONS
# =============================================================================
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

def create_bar_chart(x_var, y_var, color_var, size_var=None):
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

def create_line_chart(x_var, y_var, color_var, size_var=None):
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

def create_box_plot(x_var, y_var, color_var, size_var=None):
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

def plot_data(data):
    if sns:
        # Seaborn plot
        fig = sns.histplot(data)
    else:
        # Plotly alternative
        fig = px.histogram(data)
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Medicine Data Visualization", page_icon="📊", layout="wide")
    render_visualization_dashboard()