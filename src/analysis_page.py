import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

class AnalysisPage:
    def __init__(self):
        self.title = "Data Analysis Dashboard 📊"

    def render(self, data: pd.DataFrame):
        st.title(self.title)

        if data is None or data.empty:
            st.info("Please load data using the sidebar controls")
            return

        # Display available columns
        st.sidebar.subheader("Available Columns")
        st.sidebar.write(data.columns.tolist())

        # Let user select columns for analysis
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = data.select_dtypes(include=['object', 'bool']).columns

        tab1, tab2, tab3 = st.tabs([
            "Numerical Analysis 📈",
            "Categorical Analysis 📊",
            "Correlation Analysis 🔄"
        ])

        with tab1:
            self._render_numerical_analysis(data, numeric_cols)

        with tab2:
            self._render_categorical_analysis(data, categorical_cols)

        with tab3:
            self._render_correlation_analysis(data, numeric_cols)

    def _render_numerical_analysis(self, data, numeric_cols):
        st.subheader("Numerical Data Analysis")
        
        if len(numeric_cols) == 0:
            st.info("No numerical columns found in the dataset")
            return

        # Select column for analysis
        selected_col = st.selectbox(
            "Select column for analysis",
            numeric_cols,
            key='num_analysis'
        )

        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution plot
            try:
                fig = px.histogram(
                    data,
                    x=selected_col,
                    title=f'Distribution of {selected_col}',
                    marginal='box'
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating distribution plot: {str(e)}")

        with col2:
            # Basic statistics
            try:
                stats_df = data[selected_col].describe().round(2)
                st.dataframe(stats_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error calculating statistics: {str(e)}")

    def _render_categorical_analysis(self, data, categorical_cols):
        st.subheader("Categorical Data Analysis")
        
        if len(categorical_cols) == 0:
            st.info("No categorical columns found in the dataset")
            return

        # Select column for analysis
        selected_col = st.selectbox(
            "Select column for analysis",
            categorical_cols,
            key='cat_analysis'
        )

        try:
            # Value counts
            value_counts = data[selected_col].value_counts()
            
            # Bar chart
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Distribution of {selected_col}'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show counts table
            st.write("Value Counts:")
            st.dataframe(
                pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / len(data) * 100).round(2)
                }),
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error in categorical analysis: {str(e)}")

    def _render_correlation_analysis(self, data, numeric_cols):
        st.subheader("Correlation Analysis")
        
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numerical columns for correlation analysis")
            return

        try:
            # Correlation matrix
            corr_matrix = data[numeric_cols].corr().round(2)
            
            # Heatmap
            fig = px.imshow(
                corr_matrix,
                title='Correlation Matrix',
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Scatter plot for selected columns
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X axis", numeric_cols, key='corr_x')
            with col2:
                y_col = st.selectbox("Select Y axis", numeric_cols, key='corr_y')

            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                title=f'{x_col} vs {y_col}',
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error in correlation analysis: {str(e)}")

    def _safe_plot(self, plot_func, *args, **kwargs):
        """Safely execute a plotting function with error handling"""
        try:
            return plot_func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error creating plot: {str(e)}")
            return None