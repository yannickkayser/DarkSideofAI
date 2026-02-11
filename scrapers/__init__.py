"""
Scrapers package initialization
"""
from .base_scraper import BaseScraper
from .static_scraper import StaticScraper

__all__ = [
    'BaseScraper',
    'StaticScraper',
]
