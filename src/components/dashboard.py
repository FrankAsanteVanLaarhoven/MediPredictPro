# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime

# =============================================================================
# DASHBOARD CLASS
# =============================================================================
class Dashboard:
    def __init__(self):
        """Initialize Dashboard class with required configurations"""
        self.title = "MediPredictPro Dashboard 🏥"
        self.required_columns = ['Price', 'Manufacturer']
        self.export_timestamp = datetime.now().strftime('%Y%m%d')

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the input data
        
        Args:
            data (pd.DataFrame): Input dataframe to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if data is None or data.empty:
            st.info("Please load data using the sidebar controls")
            return False
        
        if not all(col in data.columns for col in self.required_columns):
            st.error(f"Missing required columns. Your data must include: {', '.join(self.required_columns)}")
            st.write("Available columns:", data.columns.tolist())
            st.write("Data preview:", data.head())
            return False
        return True

    def render_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Render and handle data filters
        
        Args:
            data (pd.DataFrame): Input dataframe to filter
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        try:
            with st.expander("Data Filters 🔍"):
                col1, col2 = st.columns(2)
                
                # Price Range Filter
                with col1:
                    if 'Price' in data.columns:
                        price_range = st.slider(
                            "Price Range ($)",
                            float(data['Price'].min()),
                            float(data['Price'].max()),
                            (float(data['Price'].min()), float(data['Price'].max()))
                        )
                    else:
                        st.error("Price column not found in data")
                        price_range = (0, 0)
                
                # Manufacturer Filter
                with col2:
                    if 'Manufacturer' in data.columns:
                        manufacturers = st.multiselect(
                            "Manufacturers",
                            options=sorted(data['Manufacturer'].unique()),
                            default=sorted(data['Manufacturer'].unique())
                        )
                    else:
                        st.error("Manufacturer column not found in data")
                        manufacturers = []
                
                # Apply Filters
                if 'Price' in data.columns and 'Manufacturer' in data.columns:
                    return data[
                        (data['Price'].between(price_range[0], price_range[1])) &
                        (data['Manufacturer'].isin(manufacturers))
                    ]
                return data
        except Exception as e:
            st.error(f"Error in filters: {str(e)}")
            return data

    def render_metrics(self, filtered_data: pd.DataFrame, original_data: pd.DataFrame):
        """
        Render key metrics section
        
        Args:
            filtered_data (pd.DataFrame): Filtered dataset
            original_data (pd.DataFrame): Original dataset
        """
        try:
            st.header("Key Metrics 📊")
            col1, col2, col3, col4 = st.columns(4)
            
            # Total Medicines Metric
            with col1:
                st.metric(
                    "Total Medicines",
                    f"{len(filtered_data):,}",
                    delta=f"{len(filtered_data) - len(original_data):,}"
                )
            
            # Average Price Metric
            with col2:
                if 'Price' in filtered_data.columns:
                    avg_price = filtered_data['Price'].mean()
                    st.metric(
                        "Average Price",
                        f"${avg_price:,.2f}",
                        delta=f"${avg_price - original_data['Price'].mean():,.2f}"
                    )
                else:
                    st.metric("Average Price", "N/A")
            
            # Total Value Metric
            with col3:
                if 'Price' in filtered_data.columns:
                    total_value = filtered_data['Price'].sum()
                    st.metric("Total Value", f"${total_value:,.2f}")
                else:
                    st.metric("Total Value", "N/A")
            
            # Manufacturers Count Metric
            with col4:
                if 'Manufacturer' in filtered_data.columns:
                    manufacturers_count = filtered_data['Manufacturer'].nunique()
                    st.metric("Manufacturers", manufacturers_count)
                else:
                    st.metric("Manufacturers", "N/A")
        except Exception as e:
            st.error(f"Error in metrics: {str(e)}")

    def render_charts(self, filtered_data: pd.DataFrame):
        """
        Render interactive charts section
        
        Args:
            filtered_data (pd.DataFrame): Filtered dataset to visualize
        """
        try:
            st.header("Interactive Charts 📈")
            tab1, tab2, tab3 = st.tabs(["Price Analysis", "Manufacturer Analysis", "Trends"])
            
            # Price Analysis Tab
            with tab1:
                self._render_price_analysis(filtered_data)
            
            # Manufacturer Analysis Tab
            with tab2:
                self._render_manufacturer_analysis(filtered_data)
            
            # Trends Tab
            with tab3:
                self._render_trends_analysis(filtered_data)
                
        except Exception as e:
            st.error(f"Error in charts: {str(e)}")

    def _render_price_analysis(self, data: pd.DataFrame):
        """Helper method for price analysis visualization"""
        if 'Price' in data.columns and 'Manufacturer' in data.columns:
            fig = px.histogram(
                data,
                x='Price',
                nbins=50,
                title='Price Distribution',
                color='Manufacturer'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Price or Manufacturer data not available")

    def _render_manufacturer_analysis(self, data: pd.DataFrame):
        """Helper method for manufacturer analysis visualization"""
        if 'Manufacturer' in data.columns and 'Price' in data.columns:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(
                    data,
                    names='Manufacturer',
                    title='Market Share by Manufacturer'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                avg_by_mfg = data.groupby('Manufacturer')['Price'].mean().reset_index()
                fig = px.bar(
                    avg_by_mfg,
                    x='Manufacturer',
                    y='Price',
                    title='Average Price by Manufacturer'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Manufacturer or Price data not available")

    def _render_trends_analysis(self, data: pd.DataFrame):
        """Helper method for trends analysis visualization"""
        if all(col in data.columns for col in ['Date', 'Price']):
            daily_avg = data.groupby('Date')['Price'].mean().reset_index()
            fig = px.line(
                daily_avg,
                x='Date',
                y='Price',
                title='Price Trends Over Time'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Date or Price data not available for trend analysis")

    def render_export_options(self, filtered_data: pd.DataFrame):
        """
        Render export options section
        
        Args:
            filtered_data (pd.DataFrame): Filtered dataset to export
        """
        try:
            st.header("Export Options 💾")
            col1, col2, col3 = st.columns(3)
            
            # CSV Export
            with col1:
                if st.button("Download CSV"):
                    self._handle_csv_export(filtered_data)
            
            # Excel Export
            with col2:
                if st.button("Download Excel"):
                    self._handle_excel_export(filtered_data)
            
            # JSON Export
            with col3:
                if st.button("Download JSON"):
                    self._handle_json_export(filtered_data)
                    
        except Exception as e:
            st.error(f"Error in export options: {str(e)}")

    def _handle_csv_export(self, data: pd.DataFrame):
        """Helper method for CSV export"""
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download CSV File",
            data=csv,
            file_name=f"medipredictpro_export_{self.export_timestamp}.csv",
            mime="text/csv"
        )

    def _handle_excel_export(self, data: pd.DataFrame):
        """Helper method for Excel export"""
        buffer = io.BytesIO()
        data.to_excel(buffer, index=False)
        st.download_button(
            label="Download Excel File",
            data=buffer.getvalue(),
            file_name=f"medipredictpro_export_{self.export_timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    def _handle_json_export(self, data: pd.DataFrame):
        """Helper method for JSON export"""
        json_str = data.to_json(orient='records', date_format='iso')
        st.download_button(
            label="Download JSON File",
            data=json_str,
            file_name=f"medipredictpro_export_{self.export_timestamp}.json",
            mime="application/json"
        )

    def render_dashboard(self, data: pd.DataFrame):
        """
        Main method to render the complete dashboard
        
        Args:
            data (pd.DataFrame): Input dataset to visualize
        """
        try:
            st.title(self.title)
            
            if not self.validate_data(data):
                return
            
            filtered_data = self.render_filters(data)
            self.render_metrics(filtered_data, data)
            self.render_charts(filtered_data)
            self.render_export_options(filtered_data)
            
            with st.expander("Data Preview 👀"):
                st.dataframe(filtered_data, use_container_width=True, hide_index=True)
            
            if st.checkbox("Show Advanced Analytics"):
                try:
                    render_advanced_analytics(data)
                except Exception as e:
                    st.error(f"Error in advanced analytics: {str(e)}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.exception(e)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    st.set_page_config(page_title="MediPredictPro Dashboard", page_icon="🏥", layout="wide")
    dashboard = Dashboard()
    # Add test data and rendering here if needed