"""
Logging utility for the scraping project
Provides consistent logging across all modules
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config.settings import LOGS_DIR, LOG_LEVEL, LOG_FORMAT, LOG_FILE_MAX_BYTES, LOG_FILE_BACKUP_COUNT


def setup_logger(name, log_file=None, level=None, console=True):
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional specific log file name (default: name.log)
        level: Logging level (default: from settings)
        console: Whether to also log to console (default: True)
    
    Returns:
        logging.Logger object
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level or LOG_LEVEL)
    
    formatter = logging.Formatter(LOG_FORMAT)
    
    # File handler
    if log_file is None:
        log_file = f"{name.replace('.', '_')}.log"
    
    log_path = LOGS_DIR / log_file
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT
    )
    file_handler.setLevel(level or LOG_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level or LOG_LEVEL)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name):
    """
    Get an existing logger or create a new one
    
    Args:
        name: Logger name
    
    Returns:
        logging.Logger object
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class
    Usage: class MyClass(LoggerMixin): ...
    Then access via self.logger
    """
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = get_logger(name)
        return self._logger


# Create default logger for the project
project_logger = setup_logger('scraping_project', 'scraping.log')


def log_scrape_start(url, session_id=None):
    """Log the start of a scraping operation"""
    msg = f"Starting scrape: {url}"
    if session_id:
        msg += f" (session: {session_id})"
    project_logger.info(msg)


def log_scrape_success(url, status_code, content_length, response_time):
    """Log successful scraping"""
    project_logger.info(
        f"SUCCESS: {url} | Status: {status_code} | "
        f"Size: {content_length} bytes | Time: {response_time:.2f}s"
    )


def log_scrape_error(url, error, retry_count=0):
    """Log scraping errors"""
    msg = f"ERROR: {url} | {str(error)}"
    if retry_count > 0:
        msg += f" | Retry: {retry_count}"
    project_logger.error(msg)


def log_validation_error(item_type, item_id, error):
    """Log validation errors"""
    project_logger.warning(f"VALIDATION ERROR: {item_type} {item_id} | {error}")


def log_database_operation(operation, table, record_id=None, error=None):
    """Log database operations"""
    msg = f"DB {operation}: {table}"
    if record_id:
        msg += f" (ID: {record_id})"
    
    if error:
        project_logger.error(f"{msg} | ERROR: {error}")
    else:
        project_logger.debug(msg)
