# src/components/dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def render_dashboard(data: pd.DataFrame):
    """Render the enhanced dashboard with interactive features"""
    st.title("MediPredictPro Dashboard 🏥")
    
    if data is None or data.empty:
        st.info("Please load data using the sidebar controls")
        return
    
    # Add data filters in an expander
    with st.expander("Data Filters 🔍"):
        col1, col2 = st.columns(2)
        with col1:
            # Price range filter - Add error handling for missing 'Price' column
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
        
        with col2:
            # Manufacturer filter
            if 'Manufacturer' in data.columns:
                manufacturers = st.multiselect(
                    "Manufacturers",
                    options=sorted(data['Manufacturer'].unique()),
                    default=sorted(data['Manufacturer'].unique())
                )
            else:
                st.error("Manufacturer column not found in data")
                manufacturers = []
    
    # Apply filters if columns exist
    if 'Price' in data.columns and 'Manufacturer' in data.columns:
        filtered_data = data[
            (data['Price'].between(price_range[0], price_range[1])) &
            (data['Manufacturer'].isin(manufacturers))
        ]
    else:
        filtered_data = data
    
    # Key Metrics Section
    st.header("Key Metrics 📊")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Medicines",
            f"{len(filtered_data):,}",
            delta=f"{len(filtered_data) - len(data):,}"
        )
    
    with col2:
        if 'Price' in filtered_data.columns:
            avg_price = filtered_data['Price'].mean()
            st.metric(
                "Average Price",
                f"${avg_price:,.2f}",
                delta=f"${avg_price - data['Price'].mean():,.2f}"
            )
        else:
            st.metric("Average Price", "N/A")
    
    with col3:
        if 'Price' in filtered_data.columns:
            total_value = filtered_data['Price'].sum()
            st.metric(
                "Total Value",
                f"${total_value:,.2f}"
            )
        else:
            st.metric("Total Value", "N/A")
    
    with col4:
        if 'Manufacturer' in filtered_data.columns:
            manufacturers_count = filtered_data['Manufacturer'].nunique()
            st.metric(
                "Manufacturers",
                manufacturers_count
            )
        else:
            st.metric("Manufacturers", "N/A")
    
    # Interactive Charts Section
    st.header("Interactive Charts 📈")
    
    # Tab layout for different visualizations
    tab1, tab2, tab3 = st.tabs(["Price Analysis", "Manufacturer Analysis", "Trends"])
    
    with tab1:
        if 'Price' in filtered_data.columns and 'Manufacturer' in filtered_data.columns:
            # Price Distribution
            fig = px.histogram(
                filtered_data,
                x='Price',
                nbins=50,
                title='Price Distribution',
                color='Manufacturer'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Price or Manufacturer data not available")
    
    with tab2:
        if 'Manufacturer' in filtered_data.columns and 'Price' in filtered_data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Manufacturer Market Share
                fig = px.pie(
                    filtered_data,
                    names='Manufacturer',
                    title='Market Share by Manufacturer'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average Price by Manufacturer
                avg_by_mfg = filtered_data.groupby('Manufacturer')['Price'].mean().reset_index()
                fig = px.bar(
                    avg_by_mfg,
                    x='Manufacturer',
                    y='Price',
                    title='Average Price by Manufacturer'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Manufacturer or Price data not available")
    
    with tab3:
        # Price Trends (if date column exists)
        if all(col in filtered_data.columns for col in ['Date', 'Price']):
            daily_avg = filtered_data.groupby('Date')['Price'].mean().reset_index()
            fig = px.line(
                daily_avg,
                x='Date',
                y='Price',
                title='Price Trends Over Time'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Date or Price data not available for trend analysis")
    
    # Export Options
    st.header("Export Options 💾")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Download CSV"):
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download CSV File",
                data=csv,
                file_name=f"medipredictpro_export_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Download Excel"):
            # Create Excel file in memory
            excel_buffer = filtered_data.to_excel(index=False)
            st.download_button(
                label="Download Excel File",
                data=excel_buffer,
                file_name=f"medipredictpro_export_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col3:
        if st.button("Download JSON"):
            json_str = filtered_data.to_json(orient='records', date_format='iso')
            st.download_button(
                label="Download JSON File",
                data=json_str,
                file_name=f"medipredictpro_export_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    # Data Preview
    with st.expander("Data Preview 👀"):
        st.dataframe(
            filtered_data,
            use_container_width=True,
            hide_index=True
        )