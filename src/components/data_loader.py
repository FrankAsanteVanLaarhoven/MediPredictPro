# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# =============================================================================
# LOGGER SETUP
# =============================================================================
class LoggerSetup:
    """Centralized logging system for the medicine analysis project"""
    
    def __init__(self, 
                 max_bytes: int = 10485760,  # 10MB
                 backup_count: int = 5,
                 log_level: int = logging.INFO):
        """Initialize logger setup with configuration"""
        self.log_dir = self._create_log_directory()
        self.log_file = self._create_log_file()
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.log_level = log_level
        
        # Store handlers for reuse
        self._file_handler = None
        self._console_handler = None
    
    def _create_log_directory(self) -> Path:
        """Create and return log directory"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        return log_dir
    
    def _create_log_file(self) -> Path:
        """Create and return log file path with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.log_dir / f"medicine_analysis_{timestamp}.log"
    
    def _create_file_handler(self) -> RotatingFileHandler:
        """Create and configure file handler"""
        if not self._file_handler:
            self._file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            self._file_handler.setLevel(self.log_level)
            self._file_handler.setFormatter(self._get_file_formatter())
        return self._file_handler
    
    def _create_console_handler(self) -> logging.StreamHandler:
        """Create and configure console handler"""
        if not self._console_handler:
            self._console_handler = logging.StreamHandler(sys.stdout)
            self._console_handler.setLevel(self.log_level)
            self._console_handler.setFormatter(self._get_console_formatter())
        return self._console_handler
    
    @staticmethod
    def _get_file_formatter() -> logging.Formatter:
        """Create file log formatter"""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
    
    @staticmethod
    def _get_console_formatter() -> logging.Formatter:
        """Create console log formatter"""
        return logging.Formatter(
            '%(levelname)s - %(funcName)s - %(message)s'
        )
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get configured logger instance"""
        logger = logging.getLogger(name)
        
        if not logger.handlers:
            logger.setLevel(self.log_level)
            logger.addHandler(self._create_file_handler())
            logger.addHandler(self._create_console_handler())
        
        return logger
    
    def get_log_file_path(self) -> str:
        """Return current log file path"""
        return str(self.log_file)
    
    def update_log_level(self, level: int) -> None:
        """Update logging level for all handlers"""
        self.log_level = level
        if self._file_handler:
            self._file_handler.setLevel(level)
        if self._console_handler:
            self._console_handler.setLevel(level)

# =============================================================================
# CUSTOM EXCEPTION
# =============================================================================
class CustomException(Exception):
    """Custom exception class for medicine analysis project"""
    
    ERROR_CODES = {
        1: "Data Loading Error",
        2: "Preprocessing Error",
        3: "Model Training Error",
        4: "Evaluation Error",
        5: "Visualization Error",
        6: "Database Error",
        7: "API Error",
        8: "Configuration Error",
        9: "Validation Error",
        10: "Resource Error"
    }
    
    def __init__(self, 
                 message: str, 
                 error_code: Optional[int] = None,
                 details: Optional[dict] = None):
        """Initialize custom exception with details"""
        self.message = message
        self.error_code = error_code
        self.timestamp = datetime.now()
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Format exception message"""
        error_type = self.ERROR_CODES.get(self.error_code, "Unknown Error")
        base_message = (f"[{self.timestamp}] "
                       f"{error_type} "
                       f"({self.error_code}): "
                       f"{self.message}")
        
        if self.details:
            details_str = "\nDetails:\n" + "\n".join(
                f"  {k}: {v}" for k, v in self.details.items()
            )
            return base_message + details_str
        
        return base_message
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary format"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'error_code': self.error_code,
            'error_type': self.ERROR_CODES.get(self.error_code, "Unknown Error"),
            'message': self.message,
            'details': self.details
        }

# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    # Initialize logger
    logger_setup = LoggerSetup()
    logger = logger_setup.get_logger(__name__)
    
    try:
        # Example usage
        logger.info("Starting medicine analysis")
        raise CustomException(
            message="Failed to load medicine data",
            error_code=1,
            details={
                'file': 'medicine_data.csv',
                'reason': 'File not found',
                'attempted_path': '/data/medicine_data.csv'
            }
        )
    except CustomException as e:
        logger.error(str(e))
        logger.debug(f"Exception details: {e.to_dict()}")