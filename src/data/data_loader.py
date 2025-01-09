# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
import streamlit as st
import os
import json
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# =============================================================================
# FILE HANDLER CLASS
# =============================================================================
class DataFileHandler(FileSystemEventHandler):
    """Handler for data file changes"""
    def __init__(self, data_loader):
        self.data_loader = data_loader
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.csv'):
            st.warning(f"Data file changed: {event.src_path}")
            self.data_loader.reload_data()

# =============================================================================
# DATA LOADER CLASS
# =============================================================================
class DataLoader:
    def __init__(self):
        """Initialize DataLoader with necessary configurations"""
        # Setup paths and configurations
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.sample_size = 100
        self.manufacturers = ['Pfizer', 'Novartis', 'Roche', 'Merck', 'GSK']
        
        # Initialize tracking variables
        self.last_modified = None
        self.observer = None
        self.data = None
        
        # Create data directory and setup file watching
        os.makedirs(self.data_path, exist_ok=True)
        self.setup_file_watching()

    def setup_file_watching(self):
        """Setup watchdog observer for data directory"""
        try:
            self.observer = Observer()
            self.observer.schedule(
                DataFileHandler(self),
                self.data_path,
                recursive=False
            )
            self.observer.start()
            st.sidebar.success("File watching active ✓")
        except Exception as e:
            st.sidebar.error(f"Could not setup file watching: {str(e)}")

    def reload_data(self):
        """Reload data if file has changed"""
        try:
            current_file = os.path.join(self.data_path, "medicine_details.csv")
            if os.path.exists(current_file):
                modified_time = os.path.getmtime(current_file)
                if self.last_modified != modified_time:
                    self.data = pd.read_csv(current_file)
                    self.last_modified = modified_time
                    st.session_state.data = self.data
                    st.success("Data reloaded successfully!")
        except Exception as e:
            st.error(f"Error reloading data: {str(e)}")

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
                'Is_Generic': np.random.choice([True, False], self.sample_size),
                'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Save sample data to file
            sample_file = os.path.join(self.data_path, "sample_data.csv")
            sample_data.to_csv(sample_file, index=False)
            self.last_modified = os.path.getmtime(sample_file)
            
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
            
            if os.path.exists(file_path):
                self.data = pd.read_csv(file_path)
                return self.data
            else:
                st.error(f"File not found at {file_path}")
                return None
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

    def load_uploaded_file(self, uploaded_file) -> pd.DataFrame:
        """Load data from uploaded file and save locally"""
        try:
            self.data = pd.read_csv(uploaded_file)
            
            # Save uploaded file locally for monitoring
            save_path = os.path.join(self.data_path, "uploaded_data.csv")
            self.data.to_csv(save_path, index=False)
            self.last_modified = os.path.getmtime(save_path)
            
            return self.data
            
        except Exception as e:
            st.error(f"Error loading uploaded file: {str(e)}")
            return None

    def load_kaggle_data(self, dataset_name: str, username: str = None, key: str = None) -> pd.DataFrame:
        """Load data from Kaggle"""
        try:
            if username and key:
                self._save_kaggle_credentials(username, key)
            
            api = KaggleApi()
            api.authenticate()
            
            api.dataset_download_files(
                dataset_name,
                path=self.data_path,
                unzip=True
            )
            
            self.data = pd.read_csv(os.path.join(self.data_path, "medicine_details.csv"))
            self.last_modified = os.path.getmtime(os.path.join(self.data_path, "medicine_details.csv"))
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
            
            os.chmod(f'{kaggle_path}/kaggle.json', 0o600)
            return True
        except Exception as e:
            st.error(f"Error saving Kaggle credentials: {str(e)}")
            return False

    def render_data_controls(self):
        """Render data loading controls in sidebar"""
        st.sidebar.header("Data Source")
        
        data_source = st.sidebar.radio(
            "Select Data Source",
            ["Sample Data", "Local File", "Upload File", "Kaggle Dataset"]
        )
        
        if data_source == "Sample Data":
            if st.sidebar.button("Generate Sample Data"):
                with st.spinner("Generating sample data..."):
                    st.session_state.data = self.load_sample_data()
                    if st.session_state.data is not None:
                        st.sidebar.success("✅ Sample data loaded!")
        
        elif data_source == "Local File":
            st.sidebar.write("Load from data directory:")
            if st.sidebar.button("Load Local Data"):
                with st.spinner("Loading local data..."):
                    st.session_state.data = self.load_local_file()
                    if st.session_state.data is not None:
                        st.sidebar.success("✅ Local data loaded!")
        
        elif data_source == "Upload File":
            uploaded_file = st.sidebar.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="Upload your medicine data CSV file"
            )
            if uploaded_file is not None:
                with st.spinner("Processing uploaded file..."):
                    st.session_state.data = self.load_uploaded_file(uploaded_file)
                    if st.session_state.data is not None:
                        st.sidebar.success("✅ File uploaded successfully!")
        
        elif data_source == "Kaggle Dataset":
            with st.sidebar.expander("Kaggle Settings"):
                kaggle_username = st.text_input("Kaggle Username")
                kaggle_key = st.text_input("Kaggle API Key", type="password")
                dataset_name = st.text_input(
                    "Dataset Name",
                    value="your-dataset-name/medicine-details"
                )
                
                if st.button("Load Kaggle Data"):
                    with st.spinner("Fetching data from Kaggle..."):
                        st.session_state.data = self.load_kaggle_data(
                            dataset_name,
                            kaggle_username,
                            kaggle_key
                        )
                        if st.session_state.data is not None:
                            st.sidebar.success("✅ Kaggle data loaded!")
        
        # Data Preview
        if st.session_state.data is not None:
            with st.sidebar.expander("Data Preview"):
                st.write("First few rows:")
                st.dataframe(
                    st.session_state.data.head(),
                    use_container_width=True
                )
                st.write(f"Total rows: {len(st.session_state.data):,}")

    def stop_file_watching(self):
        """Stop the file observer"""
        if self.observer:
            self.observer.stop()
            self.observer.join()

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_file_watching()

# =============================================================================
# SIDEBAR RENDERING
# =============================================================================
def render_sidebar():
    """Render the complete sidebar with file monitoring"""
    with st.sidebar:
        st.title("MediPredictPro 🏥")
        
        # Initialize DataLoader
        if 'data_loader' not in st.session_state:
            st.session_state.data_loader = DataLoader()
        
        # File monitoring status
        with st.expander("File Monitoring"):
            if st.button("Reload Data"):
                st.session_state.data_loader.reload_data()
            
            if st.button("Stop Monitoring"):
                st.session_state.data_loader.stop_file_watching()
                st.warning("File monitoring stopped")
        
        # Render data controls
        st.session_state.data_loader.render_data_controls()
        
        # Navigation
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Dashboard", "Analysis", "Predictions"],
            key="navigation"
        )
        
        return page

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    st.set_page_config(
        page_title="MediPredictPro Data Loader",
        page_icon="🏥",
        layout="wide"
    )
    render_sidebar()