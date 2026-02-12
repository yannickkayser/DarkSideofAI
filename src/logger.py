"""
Simple logging setup for the scraper
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level: str = "INFO"):
    """
    Setup logger with console and file output
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level))
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_scrape_start(logger, url: str):
    """Log scraping start"""
    logger.info(f"Starting scrape: {url}")


def log_scrape_success(logger, url: str, status_code: int, content_length: int):
    """Log successful scrape"""
    logger.info(f"✓ SUCCESS: {url} | Status: {status_code} | Size: {content_length} bytes")


def log_scrape_error(logger, url: str, error: str):
    """Log scraping error"""
    logger.error(f"✗ ERROR: {url} | {error}")


def log_validation_result(logger, valid: bool, message: str):
    """Log validation result"""
    if valid:
        logger.info(f"✓ VALIDATION PASSED: {message}")
    else:
        logger.warning(f"✗ VALIDATION FAILED: {message}")
