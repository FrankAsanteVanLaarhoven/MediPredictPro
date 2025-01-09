# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
# from dotenv import load_dotenv
# load_dotenv()
from analysis_page import AnalysisPage
from predictions_page import PredictionsPage
import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
# Try to import seaborn, use alternative if not available
try:
    import seaborn as sns
except ImportError:
    st.warning("Seaborn not available, using alternative plotting methods")
    sns = None
from scipy import stats
import io
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
import statsmodels.api as sm


# Streamlit page configuration
st.set_page_config(
    page_title="MediPredictPro 🏥",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA LOADER CLASS
# =============================================================================
class DataLoader:
    def __init__(self):
        self._load_configuration()

    def _load_configuration(self):
        self.app_name = os.environ.get("APP_NAME", "MediPredictPro")
        self.environment = os.environ.get("ENVIRONMENT", "production")
        self.default_dataset = os.environ.get("DEFAULT_DATASET", "singhnavjot2062001/11000-medicine-details")
        self.max_sample_size = int(os.environ.get("MAX_SAMPLE_SIZE", 1000))
        self.data_cache_ttl = int(os.environ.get("DATA_CACHE_TTL", 3600))
        self.debug_mode = os.environ.get("DEBUG_MODE", "false").lower() == "true"
        self.data_path = os.environ.get("DATA_PATH", "data")
        
        
        
        # Convert comma-separated string to list
        self.manufacturers = os.environ.get("MANUFACTURERS", "").split(",")
        
        # Get ranges
        self.price_range = {
            "min": int(os.environ.get("PRICE_RANGE_MIN", 10)),
            "max": int(os.environ.get("PRICE_RANGE_MAX", 1000))
        }
        
        self.effectiveness_range = {
            "min": int(os.environ.get("EFFECTIVENESS_RANGE_MIN", 70)),
            "max": int(os.environ.get("EFFECTIVENESS_RANGE_MAX", 100))
        }
        
        self.stock_range = {
            "min": int(os.environ.get("STOCK_RANGE_MIN", 0)),
            "max": int(os.environ.get("STOCK_RANGE_MAX", 1000))
        }

    def _verify_configuration(self):
        """Display current configuration settings"""
        with st.sidebar.expander("🔧 Configuration", expanded=False):
            st.write({
                "App Name": self.app_name,
                "Environment": self.environment,
                "Debug Mode": self.debug_mode,
                "Data Path": self.data_path,
                "Sample Size": self.sample_size,
                "Manufacturers": self.manufacturers,
                "Price Range": self.price_range,
                "Effectiveness Range": self.effectiveness_range,
                "Stock Range": self.stock_range
            })

    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add missing columns with default values using configured ranges"""
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        required_columns = {
            'Price': lambda x: np.random.uniform(
                self.price_range["min"],
                self.price_range["max"],
                len(x)
            ),
            'Manufacturer': lambda x: np.random.choice(self.manufacturers, len(x)),
            'Effectiveness': lambda x: np.random.uniform(
                self.effectiveness_range["min"],
                self.effectiveness_range["max"],
                len(x)
            ),
            'Stock_Level': lambda x: np.random.randint(
                self.stock_range["min"],
                self.stock_range["max"],
                len(x)
            ),
            'Is_Generic': lambda x: np.random.choice([True, False], len(x))
        }

        for col, default_func in required_columns.items():
            if col not in data.columns:
                data[col] = default_func(data)
                if self.debug_mode:
                    st.warning(f"Added missing column '{col}' with default values")

        return data

    def load_kaggle_data(self, dataset_name: str) -> pd.DataFrame:
        """Load data from Kaggle using configured settings"""
        try:
            if self.debug_mode:
                st.info(f"Loading dataset: {dataset_name}")
            
            with st.spinner("Downloading dataset..."):
                try:
                    path = kagglehub.dataset_download(dataset_name)
                    if self.debug_mode:
                        st.success(f"Download path: {path}")
                except Exception as e:
                    st.error("Failed to download dataset")
                    if self.debug_mode:
                        st.error(str(e))
                    return None

            # Find CSV files
            csv_files = []
            for root, dirs, files in os.walk(path):
                csv_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])

            if not csv_files:
                st.error("No CSV files found")
                return None

            # Select file
            selected_file = csv_files[0]
            if len(csv_files) > 1:
                file_names = [os.path.basename(f) for f in csv_files]
                selected_name = st.selectbox("Select CSV file:", file_names)
                selected_file = [f for f in csv_files if os.path.basename(f) == selected_name][0]

            # Load and validate
            try:
                data = pd.read_csv(selected_file)
                if self.debug_mode:
                    st.write("Data Preview:", data.head())
                
                self.data = self.validate_data(data)
                return self.data
            except Exception as e:
                st.error("Error loading data")
                if self.debug_mode:
                    st.error(str(e))
                return None

        except Exception as e:
            st.error("Failed to process dataset")
            if self.debug_mode:
                st.error(str(e))
            return None

    def render_data_controls(self) -> None:
        """Render data loading controls"""
    st.sidebar.header("Data Controls")
    
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Sample Data", "Local File", "Kaggle Dataset"]
    )
    
    if data_source == "Sample Data":
        if st.sidebar.button("Load Sample Data"):
            # Create sample data
            sample_data = pd.DataFrame({
                'Medicine_Name': ['Med A', 'Med B', 'Med C', 'Med D'],
                'Price': [100, 200, 150, 300],
                'Manufacturer': ['Pfizer', 'Novartis', 'Roche', 'GSK'],
                'Effectiveness': [85, 90, 75, 95],
                'Stock_Level': [500, 300, 200, 400],
                'Is_Generic': [True, False, True, False]
            })
            st.session_state.data = sample_data
            st.success("Sample data loaded successfully!")

    elif data_source == "Local File":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = self.validate_data(data)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    elif data_source == "Kaggle Dataset":
        with st.sidebar.expander("Kaggle Dataset", expanded=True):
            st.markdown("""
            ### Load Kaggle Dataset
            Enter the dataset name (format: username/dataset-name)
            
            Example:
            ```
            singhnavjot2062001/11000-medicine-details
            ```
            """)
            
            dataset_name = st.text_input(
                "Dataset Name",
                value=self.default_dataset,
                placeholder="username/dataset-name"
            )

            if st.button("Load Dataset"):
                if not dataset_name:
                    st.error("Please provide a dataset name")
                else:
                    st.session_state.data = self.load_kaggle_data(dataset_name)
                    if st.session_state.data is not None:
                        st.success("Dataset loaded successfully!")

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
        """Validate the input data"""
        if data is None or data.empty:
            st.info("Please load data using the sidebar controls")
            return False
        return True
        if not all(col in data.columns for col in self.required_columns):
            st.error(f"Missing required columns. Your data must include: {', '.join(self.required_columns)}")
            st.write("Available columns:", data.columns.tolist())
            st.write("Data preview:", data.head())
            return False
        return True

    def render_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Render and handle data filters"""
        try:
            with st.expander("Data Filters 🔍"):
                if not self.validate_data(data):
                    return data
                
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
                
                return data[
                    (data['Price'].between(price_range[0], price_range[1])) &
                    (data['Manufacturer'].isin(manufacturers))
                ]
        except Exception as e:
            st.error(f"Error in filters: {str(e)}")
            return data

    def render_metrics(self, filtered_data: pd.DataFrame, original_data: pd.DataFrame):
        """Render key metrics section"""
        try:
            st.header("Key Metrics 📊")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Medicines",
                    f"{len(filtered_data):,}",
                    delta=f"{len(filtered_data) - len(original_data):,}"
                )
            
            with col2:
                avg_price = filtered_data['Price'].mean()
                st.metric(
                    "Average Price",
                    f"${avg_price:,.2f}",
                    delta=f"${avg_price - original_data['Price'].mean():,.2f}"
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
        except Exception as e:
            st.error(f"Error in metrics: {str(e)}")

    def render_charts(self, filtered_data: pd.DataFrame):
        """Render interactive charts section"""
        try:
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
        except Exception as e:
            st.error(f"Error in charts: {str(e)}")

    def render_export_options(self, filtered_data: pd.DataFrame):
        """Render export options section"""
        try:
            st.header("Export Options 💾")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Download CSV"):
                    csv = filtered_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV File",
                        data=csv,
                        file_name=f"medipredictpro_export_{self.export_timestamp}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("Download Excel"):
                    buffer = io.BytesIO()
                    filtered_data.to_excel(buffer, index=False)
                    st.download_button(
                        label="Download Excel File",
                        data=buffer.getvalue(),
                        file_name=f"medipredictpro_export_{self.export_timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col3:
                if st.button("Download JSON"):
                    json_str = filtered_data.to_json(orient='records', date_format='iso')
                    st.download_button(
                        label="Download JSON File",
                        data=json_str,
                        file_name=f"medipredictpro_export_{self.export_timestamp}.json",
                        mime="application/json"
                    )
        except Exception as e:
            st.error(f"Error in export options: {str(e)}")

    def render_dashboard(self, data: pd.DataFrame):
        """Main method to render the complete dashboard"""
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
                render_advanced_analytics(data)
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.exception(e)

# =============================================================================
# ADVANCED ANALYTICS FUNCTIONS
# =============================================================================
def render_advanced_analytics(data: pd.DataFrame):
    """Render advanced analytics section"""
    st.header("Advanced Analytics 📊")
    
    if data is None or data.empty:
        st.info("Please load data to view analytics")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Statistical Summary",
        "Trend Analysis",
        "Correlation Analysis",
        "Outlier Detection"
    ])
    
    with tab1:
        render_statistical_summary(data)
    
    with tab2:
        render_trend_analysis(data)
    
    with tab3:
        render_correlation_analysis(data)
    
    with tab4:
        render_outlier_detection(data)

def render_statistical_summary(data: pd.DataFrame):
    """Render statistical summary tab"""
    st.subheader("Statistical Summary 📈")
    
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
            marginal='box'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    st.write("Skewness and Kurtosis Analysis")
    skew_kurt = pd.DataFrame({
        'Skewness': numeric_data.skew(),
        'Kurtosis': numeric_data.kurtosis()
    }).round(3)
    st.dataframe(skew_kurt, use_container_width=True)

def render_trend_analysis(data: pd.DataFrame):
    """Render trend analysis tab"""
    st.subheader("Trend Analysis 📈")
    
    if 'Date' in data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Moving Averages")
            selected_metric = st.selectbox(
                "Select metric for trend analysis",
                data.select_dtypes(include=['float64', 'int64']).columns
            )
            
            ma_periods = [7, 30, 90]
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data[selected_metric],
                name='Original',
                line=dict(color='gray', width=1)
            ))
            
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
            data['Month'] = pd.to_datetime(data['Date']).dt.month
            monthly_avg = data.groupby('Month')[selected_metric].mean()
            
            fig = px.line(monthly_avg, title='Monthly Seasonality Pattern')
            st.plotly_chart(fig, use_container_width=True)

def render_correlation_analysis(data: pd.DataFrame):
    """Render correlation analysis tab"""
    st.subheader("Correlation Analysis 🔄")
    
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

def render_outlier_detection(data: pd.DataFrame):
    """Render outlier detection tab"""
    st.subheader("Outlier Detection 🔍")
    
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Univariate Outlier Detection")
        selected_col = st.selectbox(
            "Select column for outlier detection",
            numeric_data.columns,
            key='outlier_col'
        )
        
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
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        outlier_detector = EllipticEnvelope(contamination=0.1, random_state=42)
        outlier_labels = outlier_detector.fit_predict(scaled_data)
        
        fig = px.scatter(
            numeric_data,
            x=numeric_data.columns[0],
            y=numeric_data.columns[1],
            color=outlier_labels.astype(str),
            title='Multivariate Outlier Detection'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(f"Number of multivariate outliers: {sum(outlier_labels == -1)}")

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def render_sidebar():
    """Render the complete sidebar"""
    with st.sidebar:
        st.title(os.environ.get("APP_NAME", "MediPredictPro 🏥"))
        
        data_loader = DataLoader()
        data_loader.render_data_controls()
        
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Dashboard", "Analysis", "Predictions"],
            key="navigation"
        )
        
        return page
    
def render_kaggle_instructions():
    """Render instructions for getting Kaggle credentials"""
    st.markdown("""
    ### How to get Kaggle credentials:
    1. Sign up for a Kaggle account at [kaggle.com](https://www.kaggle.com)
    2. Go to your account settings (click on your profile picture → Account)
    3. Scroll down to the API section
    4. Click "Create New API Token"
    5. This will download a `kaggle.json` file containing your credentials
    6. Enter the username and key from that file in the fields above
    """)

# Add health check function
@st.cache_data
def health_check():
    """API endpoint for health checking"""
    return json.dumps({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

# healtcheck
if "health" in st.query_params:
    st.write(health_check())
    st.stop()
    
    
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    st.warning("Kaggle API not available, using alternative data source")
    # Use alternative data loading method

def main():
    try:
        if 'data' not in st.session_state:
            st.session_state.data = None
        
        if 'dashboard' not in st.session_state:
            st.session_state.dashboard = Dashboard()
        
        current_page = render_sidebar()
        
        if current_page == "Dashboard":
            st.session_state.dashboard.render_dashboard(st.session_state.data)
        elif current_page == "Analysis":
            analysis_page = AnalysisPage()
            analysis_page.render(st.session_state.data)
        elif current_page == "Predictions":
            predictions_page = PredictionsPage()
            predictions_page.render(st.session_state.data)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()