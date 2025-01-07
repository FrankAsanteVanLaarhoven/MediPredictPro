import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
from datetime import datetime

class LoggerSetup:
    """Centralized logging system for the medicine analysis project"""
    
    def __init__(self, max_bytes=10485760, backup_count=5):
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"medicine_analysis_{timestamp}.log"
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
    def get_logger(self, name):
        """Get logger instance with specified configuration"""
        logger = logging.getLogger(name)
        
        # Avoid adding handlers multiple times
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            
            # Rotating file handler
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(levelname)s - %(funcName)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            
            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def get_log_file_path(self):
        """Return the current log file path"""
        return str(self.log_file)

class CustomException(Exception):
    """Custom exception class for medicine analysis project"""
    
    ERROR_CODES = {
        1: "Data Loading Error",
        2: "Preprocessing Error",
        3: "Model Training Error",
        4: "Evaluation Error",
        5: "Visualization Error"
    }
    
    def __init__(self, message, error_code=None):
        self.message = message
        self.error_code = error_code
        self.timestamp = datetime.now()
        super().__init__(self.message)
    
    def __str__(self):
        error_type = self.ERROR_CODES.get(self.error_code, "Unknown Error")
        if self.error_code:
            return f"[{self.timestamp}] {error_type} ({self.error_code}): {self.message}"
        return f"[{self.timestamp}] Error: {self.message}"
