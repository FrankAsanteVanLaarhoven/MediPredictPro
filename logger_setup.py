import logging
import sys
from pathlib import Path
from datetime import datetime
import streamlit as st

class LoggerSetup:
    def __init__(self):
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"medicine_analysis_{timestamp}.log"
        
        # Initialize streamlit state for logging
        if 'logger_initialized' not in st.session_state:
            st.session_state.logger_initialized = False
    
    def get_logger(self, name):
        try:
            logger = logging.getLogger(name)
            
            if not logger.handlers and not st.session_state.logger_initialized:
                logger.setLevel(logging.INFO)
                
                # File handler with error handling
                try:
                    file_handler = logging.FileHandler(self.log_file)
                    file_handler.setLevel(logging.INFO)
                    file_formatter = logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                    file_handler.setFormatter(file_formatter)
                    logger.addHandler(file_handler)
                except Exception as e:
                    st.error(f"Failed to setup file logging: {str(e)}")
                
                # Console handler
                try:
                    console_handler = logging.StreamHandler(sys.stdout)
                    console_handler.setLevel(logging.INFO)
                    console_formatter = logging.Formatter(
                        '%(levelname)s - %(message)s'
                    )
                    console_handler.setFormatter(console_formatter)
                    logger.addHandler(console_handler)
                except Exception as e:
                    st.error(f"Failed to setup console logging: {str(e)}")
                
                # Streamlit handler for UI feedback
                class StreamlitHandler(logging.Handler):
                    def emit(self, record):
                        try:
                            msg = self.format(record)
                            if record.levelno >= logging.ERROR:
                                st.error(msg)
                            elif record.levelno >= logging.WARNING:
                                st.warning(msg)
                            elif record.levelno >= logging.INFO:
                                st.info(msg)
                        except Exception:
                            pass
                
                streamlit_handler = StreamlitHandler()
                streamlit_handler.setLevel(logging.INFO)
                streamlit_formatter = logging.Formatter('%(message)s')
                streamlit_handler.setFormatter(streamlit_formatter)
                logger.addHandler(streamlit_handler)
                
                st.session_state.logger_initialized = True
            
            return logger
            
        except Exception as e:
            st.error(f"Failed to initialize logger: {str(e)}")
            return logging.getLogger(name)  # Return basic logger as fallback

def init_logging():
    """Initialize logging for the application"""
    try:
        logger_setup = LoggerSetup()
        logger = logger_setup.get_logger(__name__)
        logger.info("Logging initialized successfully")
        return logger
    except Exception as e:
        st.error(f"Failed to initialize logging: {str(e)}")
        return logging.getLogger(__name__)  # Return basic logger as fallback
