import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(logger_name="NiftyQuantApp"):
    """Creates a production-grade logger."""
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is called multiple times
    if not logger.handlers:
        # 1. File Handler (Keeps file size under 5MB, keeps 3 backups)
        file_handler = RotatingFileHandler('logs/app.log', maxBytes=5000000, backupCount=3)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # 2. Console Handler (For terminal output)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger