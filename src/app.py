from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np  # Added numpy import
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
import seaborn as sns
from scipy import stats
from data.data_loader import DataLoader, render_sidebar 
import atexit
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope

# Must be the first Streamlit command
st.set_page_config(
    page_title="MediPredictPro 🏥",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DataLoader:
    """Class to handle all data loading operations"""
    
    def __init__(self):
        self.data = None
        self.data_path = "data"
        self.sample_size = 100
        self.manufacturers = ['Pfizer', 'Novartis', 'Roche', 'Merck', 'GSK']
        
        # Ensure data directory exists
        os.makedirs(self.data_path, exist_ok=True)
    
    def load_sample_data(self) -> pd.DataFrame:
        """Generate and load sample data for testing"""
        try:
            sample_data = pd.DataFrame({
                'Medicine_Name': [f'Medicine_{i}' for i in range(self.sample_size)],
                'Price': np.random.uniform(10, 1000, self.sample_size),
                'Manufacturer': np.random.choice(self.manufacturers, self.sample_size),
                'Date': pd.date_range(start='2023-01-01', periods=self.sample_size, freq='D'),
                'Effectiveness': np.random.uniform(70, 100, self.sample_size),
                'Stock_Level': np.random.randint(0, 1000, self.sample_size),
                'Is_Generic': np.random.choice([True, False], self.sample_size)
            })
            
            self.data = sample_data
            return self.data
            
        except Exception as e:
            st.error(f"Error generating sample data: {str(e)}")
            return None
    
    def load_local_file(self, file_path: str = None) -> pd.DataFrame:
        """Load data from local CSV file"""
        try:
            if file_path is None:
                file_path = os.path.join(self.data_path, "medicine_details.csv")
            
            self.data = pd.read_csv(file_path)
            return self.data
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    def load_uploaded_file(self, uploaded_file) -> pd.DataFrame:
        """Load data from uploaded file"""
        try:
            self.data = pd.read_csv(uploaded_file)
            return self.data
            
        except Exception as e:
            st.error(f"Error loading uploaded file: {str(e)}")
            return None
    
    def load_kaggle_data(self, dataset_name: str, username: str = None, key: str = None) -> pd.DataFrame:
        """Load data from Kaggle"""
        try:
            # Save credentials if provided
            if username and key:
                self._save_kaggle_credentials(username, key)
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Download the dataset
            api.dataset_download_files(
                dataset_name,
                path=self.data_path,
                unzip=True
            )
            
            # Load the downloaded CSV
            self.data = pd.read_csv(os.path.join(self.data_path, "medicine_details.csv"))
            return self.data
            
        except Exception as e:
            st.error(f"Error loading from Kaggle: {str(e)}")
            return None
    
    def _save_kaggle_credentials(self, username: str, key: str) -> bool:
        """Save Kaggle credentials to config file"""
        try:
            kaggle_path = os.path.expanduser('~/.kaggle')
            os.makedirs(kaggle_path, exist_ok=True)
            
            with open(f'{kaggle_path}/kaggle.json', 'w') as f:
                json.dump({
                    "username": username,
                    "key": key
                }, f)
            
            # Set file permissions
            os.chmod(f'{kaggle_path}/kaggle.json', 0o600)
            return True
            
        except Exception as e:
            st.error(f"Error saving Kaggle credentials: {str(e)}")
            return False
    
    def render_data_controls(self) -> None:
        """Render data loading controls in sidebar"""
        st.sidebar.header("Data Controls")
        
        data_source = st.sidebar.radio(
            "Select Data Source",
            ["Sample Data", "Local File", "Kaggle Dataset"]
        )
        
        if data_source == "Sample Data":
            if st.sidebar.button("Load Sample Data"):
                st.session_state.data = self.load_sample_data()
                if st.session_state.data is not None:
                    st.sidebar.success("Sample data loaded successfully!")
        
        elif data_source == "Local File":
            uploaded_file = st.sidebar.file_uploader(
                "Upload CSV file",
                type=['csv']
            )
            if uploaded_file is not None:
                st.session_state.data = self.load_uploaded_file(uploaded_file)
                if st.session_state.data is not None:
                    st.sidebar.success("File uploaded successfully!")
        
        elif data_source == "Kaggle Dataset":
            with st.sidebar.expander("Kaggle Credentials"):
                kaggle_username = st.text_input("Kaggle Username")
                kaggle_key = st.text_input("Kaggle API Key", type="password")
                
                if st.button("Save Credentials"):
                    if self._save_kaggle_credentials(kaggle_username, kaggle_key):
                        st.success("Kaggle credentials saved!")
            
            dataset_name = st.sidebar.text_input(
                "Kaggle Dataset Name",
                value="your-dataset-name/medicine-details"
            )
            
            if st.sidebar.button("Load Kaggle Data"):
                st.session_state.data = self.load_kaggle_data(dataset_name)
                if st.session_state.data is not None:
                    st.sidebar.success("Kaggle data loaded successfully!")

def render_sidebar():
    """Render the complete sidebar"""
    with st.sidebar:
        st.title("MediPredictPro 🏥")
        
        # Initialize DataLoader
        data_loader = DataLoader()
        
        # Render data controls
        data_loader.render_data_controls()
        
        # Navigation
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Dashboard", "Analysis", "Predictions"],
            key="navigation"
        )
        
        return page
# In render_advanced_analytics function, update the skewness calculation


def render_advanced_analytics(data: pd.DataFrame):
    """Render advanced analytics section"""
    st.header("Advanced Analytics 📊")
    
    if data is None or data.empty:
        st.info("Please load data to view analytics")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Statistical Summary",
        "Trend Analysis",
        "Correlation Analysis",
        "Outlier Detection"
    ])
    
    # 1. Statistical Summary Tab
    with tab1:
        st.subheader("Statistical Summary 📈")
        
        # Numerical summary
        col1, col2 = st.columns(2)
        with col1:
            st.write("Numerical Statistics")
            stats_df = data.describe().round(2)
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.write("Distribution Analysis")
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            selected_col = st.selectbox("Select column for distribution", numeric_cols)
            
            fig = px.histogram(
                data,
                x=selected_col,
                title=f'Distribution of {selected_col}',
                marginal='box'  # adds a box plot above the histogram
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Skewness and Kurtosis
        st.write("Skewness and Kurtosis Analysis")
        skew_kurt = pd.DataFrame({
            'Skewness': data.skew(),
            'Kurtosis': data.kurtosis()
        }).round(3)
        st.dataframe(skew_kurt, use_container_width=True)
    
    # 2. Trend Analysis Tab
    with tab2:
        st.subheader("Trend Analysis 📈")
        
        if 'Date' in data.columns:
            # Time series decomposition
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Moving Averages")
                selected_metric = st.selectbox(
                    "Select metric for trend analysis",
                    data.select_dtypes(include=['float64', 'int64']).columns
                )
                
                # Calculate moving averages
                ma_periods = [7, 30, 90]
                fig = go.Figure()
                
                # Add original data
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data[selected_metric],
                    name='Original',
                    line=dict(color='gray', width=1)
                ))
                
                # Add moving averages
                colors = ['blue', 'green', 'red']
                for period, color in zip(ma_periods, colors):
                    ma = data[selected_metric].rolling(window=period).mean()
                    fig.add_trace(go.Scatter(
                        x=data['Date'],
                        y=ma,
                        name=f'{period}-day MA',
                        line=dict(color=color, width=2)
                    ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("Seasonality Analysis")
                # Group by month/day for seasonality
                data['Month'] = data['Date'].dt.month
                monthly_avg = data.groupby('Month')[selected_metric].mean()
                
                fig = px.line(
                    monthly_avg,
                    title='Monthly Seasonality Pattern'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # 3. Correlation Analysis Tab
    with tab3:
        st.subheader("Correlation Analysis 🔄")
        
        # Get numeric columns for correlation
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Correlation Matrix")
            corr_matrix = numeric_data.corr().round(2)
            
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu',
                aspect='auto',
                title='Correlation Heatmap'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("Scatter Plot Analysis")
            x_col = st.selectbox("Select X axis", numeric_data.columns, key='x_axis')
            y_col = st.selectbox("Select Y axis", numeric_data.columns, key='y_axis')
            
            fig = px.scatter(
                numeric_data,
                x=x_col,
                y=y_col,
                trendline="ols",
                title=f'{x_col} vs {y_col}'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 4. Outlier Detection Tab
    with tab4:
        st.subheader("Outlier Detection 🔍")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Univariate Outlier Detection")
            selected_col = st.selectbox(
                "Select column for outlier detection",
                numeric_data.columns,
                key='outlier_col'
            )
            
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(data[selected_col]))
            outliers = data[z_scores > 3]
            
            fig = px.box(
                data,
                y=selected_col,
                title=f'Box Plot with Outliers for {selected_col}'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"Number of outliers detected: {len(outliers)}")
        
        with col2:
            st.write("Multivariate Outlier Detection")
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # Detect outliers using Elliptic Envelope
            outlier_detector = EllipticEnvelope(contamination=0.1, random_state=42)
            outlier_labels = outlier_detector.fit_predict(scaled_data)
            
            # Plot results
            fig = px.scatter(
                numeric_data,
                x=numeric_data.columns[0],
                y=numeric_data.columns[1],
                color=outlier_labels.astype(str),
                title='Multivariate Outlier Detection'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"Number of multivariate outliers: {sum(outlier_labels == -1)}")
            
    # Get only numeric columns for skewness and kurtosis
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    # Update skewness and kurtosis calculation
    st.write("Skewness and Kurtosis Analysis")
    skew_kurt = pd.DataFrame({
        'Skewness': numeric_data.skew(),
        'Kurtosis': numeric_data.kurtosis()
    }).round(3)
    st.dataframe(skew_kurt, use_container_width=True)
            
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
            price_range = st.slider(
                "Price Range ($)",
                float(data['Price'].min()),
                float(data['Price'].max()),
                (float(data['Price'].min()), float(data['Price'].max()))
            )
        with col2:
            manufacturers = st.multiselect(
                "Manufacturers",
                options=sorted(data['Manufacturer'].unique()),
                default=sorted(data['Manufacturer'].unique())
            )
        
        filtered_data = data[
            (data['Price'].between(price_range[0], price_range[1])) &
            (data['Manufacturer'].isin(manufacturers))
        ]
    
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
        avg_price = filtered_data['Price'].mean()
        st.metric(
            "Average Price",
            f"${avg_price:,.2f}",
            delta=f"${avg_price - data['Price'].mean():,.2f}"
        )
    
    with col3:
        total_value = filtered_data['Price'].sum()
        st.metric(
            "Total Value",
            f"${total_value:,.2f}"
        )
    
    with col4:
        manufacturers_count = filtered_data['Manufacturer'].nunique()
        st.metric(
            "Manufacturers",
            manufacturers_count
        )
    
    # Interactive Charts Section
    st.header("Interactive Charts 📈")
    tab1, tab2, tab3 = st.tabs(["Price Analysis", "Manufacturer Analysis", "Trends"])
    
    with tab1:
        fig = px.histogram(
            filtered_data,
            x='Price',
            nbins=50,
            title='Price Distribution',
            color='Manufacturer'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                filtered_data,
                names='Manufacturer',
                title='Market Share by Manufacturer'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_by_mfg = filtered_data.groupby('Manufacturer')['Price'].mean().reset_index()
            fig = px.bar(
                avg_by_mfg,
                x='Manufacturer',
                y='Price',
                title='Average Price by Manufacturer'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if 'Date' in filtered_data.columns:
            daily_avg = filtered_data.groupby('Date')['Price'].mean().reset_index()
            fig = px.line(
                daily_avg,
                x='Date',
                y='Price',
                title='Price Trends Over Time'
            )
            st.plotly_chart(fig, use_container_width=True)
    
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
    
    # Add Advanced Analytics section
    if st.checkbox("Show Advanced Analytics"):
        render_advanced_analytics(data)



def render_analysis(data):
    """Placeholder for analysis page"""
    st.header("Analysis")
    if data is not None:
        st.write("Analysis coming soon...")
    else:
        st.info("Please load data using the sidebar controls")

def render_predictions(data):
    """Placeholder for predictions page"""
    st.header("Predictions")
    if data is not None:
        st.write("Predictions coming soon...")
    else:
        st.info("Please load data using the sidebar controls")

from src.data.data_loader import DataLoader, render_sidebar

def main():
    """Main application function"""
    try:
        # Initialize session state
        if 'data' not in st.session_state:
            st.session_state.data = None
        
        # Render sidebar and get current page
        current_page = render_sidebar()
        
        # Register cleanup function
        atexit.register(lambda: st.session_state.data_loader.stop_file_watching() 
                       if 'data_loader' in st.session_state else None)
        # Render the appropriate page
        if current_page == "Dashboard":
            render_dashboard(st.session_state.data)
        elif current_page == "Analysis":
            render_analysis(st.session_state.data)
        elif current_page == "Predictions":
            render_predictions(st.session_state.data)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()