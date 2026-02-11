"""
Base scraper class with common functionality
All specific scrapers inherit from this
"""
import time
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime
from urllib.parse import urljoin, urlparse

from config.settings import RATE_LIMIT_DELAY, RETRY_ATTEMPTS, RETRY_DELAY, USER_AGENT
from utils.logger import get_logger, log_scrape_start, log_scrape_success, log_scrape_error
from utils.validation import (
    validate_url, normalize_url, should_scrape_url,
    validate_content, validate_page_data
)
from utils.database import DatabaseManager
from utils.file_handler import FileHandler
from utils.sitemap_parser import (
    prepare_url_queue_from_sitemap, 
    get_recommended_rate_limit,
    check_robots_and_sitemap
)


class BaseScraper(ABC):
    """
    Abstract base class for all scrapers
    Provides common functionality and interface
    """
    
    def __init__(self, config: Dict[str, Any], db_path: str):
        """
        Initialize base scraper
        
        Args:
            config: Scraper configuration dictionary
            db_path: Path to database
        """
        self.config = config
        self.db_path = db_path
        self.logger = get_logger(self.__class__.__name__)
        
        # Extract common config values
        self.base_url = config['base_url']
        self.domain = config.get('allowed_domains', [])[0] if config.get('allowed_domains') else None
        self.name = config.get('name', self.domain)
        self.max_depth = config.get('max_depth', 3)
        self.rate_limit = config.get('rate_limit', RATE_LIMIT_DELAY)
        
        # Sitemap support
        self.use_sitemap = config.get('use_sitemap', True)  # Default to True
        self.sitemap_urls = []
        
        # URL patterns
        self.include_patterns = config.get('include_patterns', [])
        self.exclude_patterns = config.get('exclude_patterns', [])
        
        # Selectors
        self.selectors = config.get('selectors', {})
        
        # Page classification
        self.page_types = config.get('page_types', {})
        
        # Utilities
        self.file_handler = FileHandler()
        
        # Tracking
        self.visited_urls = set()
        self.failed_urls = set()
        self.last_request_time = 0
        
        # Statistics
        self.stats = {
            'pages_scraped': 0,
            'pages_failed': 0,
            'total_bytes': 0,
            'start_time': None,
            'end_time': None
        }
    
    def check_robots_and_get_sitemap(self) -> Optional[List[str]]:
        """
        Check robots.txt and get URLs from sitemap if available
        
        Returns:
            List of URLs from sitemap or None
        """
        if not self.use_sitemap:
            self.logger.info("Sitemap usage disabled in config")
            return None
        
        try:
            self.logger.info(f"Checking robots.txt and sitemap for {self.base_url}")
            
            # Get recommended crawl delay from robots.txt
            recommended_delay = get_recommended_rate_limit(self.base_url, self.rate_limit)
            
            if recommended_delay != self.rate_limit:
                self.logger.info(f"Updating rate limit from {self.rate_limit}s to {recommended_delay}s (from robots.txt)")
                self.rate_limit = recommended_delay
            
            # Get URLs from sitemap
            urls = prepare_url_queue_from_sitemap(
                self.base_url,
                self.include_patterns,
                self.exclude_patterns
            )
            
            if urls:
                self.logger.info(f"Found {len(urls)} URLs in sitemap")
                self.sitemap_urls = urls
                return urls
            else:
                self.logger.warning("No URLs found in sitemap, will use crawling")
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing sitemap: {e}")
            return None
    
    def _respect_rate_limit(self):
        """Ensure rate limiting between requests"""
        if self.last_request_time > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit:
                sleep_time = self.rate_limit - elapsed
                self.logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _should_scrape_url(self, url: str) -> bool:
        """
        Check if URL should be scraped
        
        Args:
            url: URL to check
        
        Returns:
            bool: True if should scrape
        """
        # Normalize URL
        url = normalize_url(url, self.base_url)
        
        # Check if already visited
        if url in self.visited_urls:
            self.logger.debug(f"Already visited: {url}")
            return False
        
        # Check if previously failed
        if url in self.failed_urls:
            self.logger.debug(f"Previously failed: {url}")
            return False
        
        # Validate URL
        validation = validate_url(url, self.config.get('allowed_domains'))
        if not validation['is_valid']:
            self.logger.debug(f"Invalid URL: {url} - {validation['errors']}")
            return False
        
        # Check patterns
        if not should_scrape_url(url, self.include_patterns, self.exclude_patterns):
            self.logger.debug(f"URL doesn't match patterns: {url}")
            return False
        
        return True
    
    def _classify_page(self, url: str) -> Optional[str]:
        """
        Classify page based on URL patterns
        
        Args:
            url: Page URL
        
        Returns:
            Page type or None
        """
        for page_type, patterns in self.page_types.items():
            for pattern in patterns:
                import re
                if re.search(pattern, url, re.IGNORECASE):
                    return page_type
        return None
    
    def _extract_links(self, soup, base_url: str) -> List[Dict[str, str]]:
        """
        Extract links from BeautifulSoup object
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links
        
        Returns:
            List of link dictionaries
        """
        links = []
        link_selector = self.selectors.get('links', 'a[href]')
        
        for link_elem in soup.select(link_selector):
            href = link_elem.get('href', '').strip()
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
            
            # Normalize URL
            full_url = normalize_url(urljoin(base_url, href))
            
            # Determine link type
            parsed_base = urlparse(base_url)
            parsed_link = urlparse(full_url)
            
            if parsed_link.netloc == parsed_base.netloc:
                link_type = 'internal'
            else:
                link_type = 'external'
            
            links.append({
                'url': full_url,
                'anchor_text': link_elem.get_text(strip=True),
                'link_type': link_type
            })
        
        return links
    
    def _extract_images(self, soup, base_url: str) -> List[Dict[str, Any]]:
        """
        Extract images from BeautifulSoup object
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative URLs
        
        Returns:
            List of image dictionaries
        """
        images = []
        image_selector = self.selectors.get('images', 'img[src]')
        
        for img in soup.select(image_selector):
            src = img.get('src', '').strip()
            if not src:
                continue
            
            # Normalize URL
            full_url = normalize_url(urljoin(base_url, src))
            
            images.append({
                'url': full_url,
                'alt_text': img.get('alt', ''),
                'width': img.get('width'),
                'height': img.get('height')
            })
        
        return images
    
    @abstractmethod
    def scrape_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a single page (must be implemented by subclasses)
        
        Args:
            url: URL to scrape
        
        Returns:
            Dictionary with page data or None if failed
        """
        pass
    
    @abstractmethod
    def scrape_website(self, start_url: Optional[str] = None, 
                      max_pages: Optional[int] = None) -> Dict[str, Any]:
        """
        Scrape entire website (must be implemented by subclasses)
        
        Args:
            start_url: Starting URL (defaults to base_url)
            max_pages: Maximum number of pages to scrape
        
        Returns:
            Dictionary with scraping results
        """
        pass
    
    def validate_scraped_data(self, page_data: Dict[str, Any]) -> bool:
        """
        Validate scraped page data
        
        Args:
            page_data: Dictionary with scraped data
        
        Returns:
            bool: True if valid
        """
        validation = validate_page_data(page_data)
        
        if not validation['is_valid']:
            self.logger.warning(
                f"Validation failed for {page_data.get('url', 'unknown')}: "
                f"{validation['errors']}"
            )
            return False
        
        return True
    
    def save_page(self, page_data: Dict[str, Any], website_id: int, 
                 session_id: int) -> Optional[int]:
        """
        Save page data to database and file
        
        Args:
            page_data: Dictionary with page data
            website_id: Website ID
            session_id: Scrape session ID
        
        Returns:
            Page ID or None if failed
        """
        try:
            # Save content to JSON file
            content_file = self.file_handler.save_page_content(
                page_data['url'],
                page_data,
                self.domain
            )
            
            # Save to database
            with DatabaseManager(self.db_path) as db:
                page = db.create_page(
                    website_id=website_id,
                    scrape_session_id=session_id,
                    url=page_data['url'],
                    title=page_data.get('title'),
                    description=page_data.get('description'),
                    page_type=page_data.get('page_type'),
                    text_content=page_data.get('text_content'),
                    content_file=content_file,
                    status_code=page_data.get('status_code'),
                    content_type=page_data.get('content_type'),
                    content_length=page_data.get('content_length'),
                    response_time=page_data.get('response_time'),
                    is_successful=page_data.get('is_successful', True),
                    links_count=page_data.get('links_count', 0),
                    images_count=page_data.get('images_count', 0),
                    depth=page_data.get('depth', 0)
                )
                
                return page.id
                
        except Exception as e:
            self.logger.error(f"Error saving page {page_data.get('url')}: {e}")
            return None
    
    def print_progress(self):
        """Print scraping progress - feedback function"""
        print(f"\rPages: {self.stats['pages_scraped']} | "
              f"Failed: {self.stats['pages_failed']} | "
              f"Queue: {len(self.visited_urls)}", end='', flush=True)
    
    def print_summary(self):
        """Print scraping summary - feedback function"""
        duration = 0
        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        print("\n\n" + "="*60)
        print("SCRAPING SUMMARY")
        print("="*60)
        print(f"Website: {self.name}")
        print(f"Pages Scraped: {self.stats['pages_scraped']}")
        print(f"Pages Failed: {self.stats['pages_failed']}")
        print(f"Success Rate: {(self.stats['pages_scraped']/(self.stats['pages_scraped']+self.stats['pages_failed'])*100) if (self.stats['pages_scraped']+self.stats['pages_failed']) > 0 else 0:.1f}%")
        print(f"Total Data: {self.stats['total_bytes'] / 1024 / 1024:.2f} MB")
        print(f"Duration: {duration:.1f}s")
        print(f"Pages/second: {self.stats['pages_scraped']/duration:.2f}" if duration > 0 else "Pages/second: N/A")
        print("="*60 + "\n")
