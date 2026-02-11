"""
Utilities package initialization
"""
from .logger import setup_logger, get_logger, LoggerMixin
from .validation import (
    validate_url, is_valid_url, normalize_url,
    validate_content, validate_page_data,
    matches_pattern, should_scrape_url,
    sanitize_filename, check_validation_summary
)
from .database import DatabaseManager, check_database_status
from .file_handler import FileHandler, check_file_structure
from .sitemap_parser import (
    SitemapParser, RobotsParser, SitemapBasedScraper,
    check_robots_and_sitemap, prepare_url_queue_from_sitemap,
    get_recommended_rate_limit
)

__all__ = [
    'setup_logger',
    'get_logger',
    'LoggerMixin',
    'validate_url',
    'is_valid_url',
    'normalize_url',
    'validate_content',
    'validate_page_data',
    'matches_pattern',
    'should_scrape_url',
    'sanitize_filename',
    'check_validation_summary',
    'DatabaseManager',
    'check_database_status',
    'FileHandler',
    'check_file_structure',
    'SitemapParser',
    'RobotsParser',
    'SitemapBasedScraper',
    'check_robots_and_sitemap',
    'prepare_url_queue_from_sitemap',
    'get_recommended_rate_limit',
]
