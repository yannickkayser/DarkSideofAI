"""
Sitemap and robots.txt utilities for efficient web scraping
"""
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from datetime import datetime

from config.settings import USER_AGENT, REQUEST_TIMEOUT
from utils.logger import get_logger

logger = get_logger(__name__)


class SitemapParser:
    """
    Parse XML sitemaps to get all URLs from a website
    Supports sitemap index files and regular sitemaps
    """
    
    def __init__(self, base_url: str):
        """
        Initialize sitemap parser
        
        Args:
            base_url: Base URL of the website
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        
        # Common sitemap locations
        self.common_sitemap_paths = [
            '/sitemap.xml',
            '/sitemap_index.xml',
            '/sitemap1.xml',
            '/post-sitemap.xml',
            '/page-sitemap.xml',
        ]
    
    def find_sitemap_url(self, robots_txt_content: Optional[str] = None) -> Optional[str]:
        """
        Find sitemap URL from robots.txt or common locations
        
        Args:
            robots_txt_content: Content of robots.txt (optional)
        
        Returns:
            Sitemap URL or None
        """
        # First try to get from robots.txt
        if robots_txt_content:
            for line in robots_txt_content.split('\n'):
                if line.lower().startswith('sitemap:'):
                    sitemap_url = line.split(':', 1)[1].strip()
                    logger.info(f"Found sitemap in robots.txt: {sitemap_url}")
                    return sitemap_url
        
        # Try common locations
        for path in self.common_sitemap_paths:
            sitemap_url = self.base_url + path
            try:
                response = self.session.head(sitemap_url, timeout=REQUEST_TIMEOUT)
                if response.status_code == 200:
                    logger.info(f"Found sitemap at: {sitemap_url}")
                    return sitemap_url
            except Exception as e:
                logger.debug(f"No sitemap at {sitemap_url}: {e}")
                continue
        
        logger.warning(f"No sitemap found for {self.base_url}")
        return None
    
    def fetch_sitemap(self, sitemap_url: str) -> Optional[str]:
        """
        Fetch sitemap content
        
        Args:
            sitemap_url: URL of the sitemap
        
        Returns:
            Sitemap XML content or None
        """
        try:
            response = self.session.get(sitemap_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            logger.info(f"Fetched sitemap: {sitemap_url}")
            return response.text
        except Exception as e:
            logger.error(f"Error fetching sitemap {sitemap_url}: {e}")
            return None
    
    def parse_sitemap(self, xml_content: str) -> List[Dict[str, any]]:
        """
        Parse sitemap XML and extract URLs
        
        Args:
            xml_content: XML content of sitemap
        
        Returns:
            List of URL dictionaries with metadata
        """
        urls = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Handle namespace
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
            # Check if it's a sitemap index or regular sitemap
            if root.tag.endswith('sitemapindex'):
                # It's an index, get all sitemap URLs
                logger.info("Parsing sitemap index")
                sitemap_urls = []
                
                for sitemap in root.findall('ns:sitemap', namespace):
                    loc = sitemap.find('ns:loc', namespace)
                    if loc is not None and loc.text:
                        sitemap_urls.append(loc.text)
                
                # Recursively fetch and parse each sitemap
                for sitemap_url in sitemap_urls:
                    logger.info(f"Fetching nested sitemap: {sitemap_url}")
                    nested_content = self.fetch_sitemap(sitemap_url)
                    if nested_content:
                        urls.extend(self.parse_sitemap(nested_content))
            
            else:
                # Regular sitemap with URLs
                logger.info("Parsing regular sitemap")
                
                for url_elem in root.findall('ns:url', namespace):
                    loc = url_elem.find('ns:loc', namespace)
                    if loc is not None and loc.text:
                        url_data = {'url': loc.text}
                        
                        # Extract optional metadata
                        lastmod = url_elem.find('ns:lastmod', namespace)
                        if lastmod is not None and lastmod.text:
                            url_data['lastmod'] = lastmod.text
                        
                        priority = url_elem.find('ns:priority', namespace)
                        if priority is not None and priority.text:
                            url_data['priority'] = float(priority.text)
                        
                        changefreq = url_elem.find('ns:changefreq', namespace)
                        if changefreq is not None and changefreq.text:
                            url_data['changefreq'] = changefreq.text
                        
                        urls.append(url_data)
        
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
        except Exception as e:
            logger.error(f"Error parsing sitemap: {e}")
        
        logger.info(f"Extracted {len(urls)} URLs from sitemap")
        return urls
    
    def get_all_urls(self, robots_txt_content: Optional[str] = None) -> List[Dict[str, any]]:
        """
        Get all URLs from sitemap(s)
        
        Args:
            robots_txt_content: Content of robots.txt (optional)
        
        Returns:
            List of URL dictionaries
        """
        sitemap_url = self.find_sitemap_url(robots_txt_content)
        
        if not sitemap_url:
            return []
        
        xml_content = self.fetch_sitemap(sitemap_url)
        
        if not xml_content:
            return []
        
        return self.parse_sitemap(xml_content)


class RobotsParser:
    """
    Parse and respect robots.txt rules
    """
    
    def __init__(self, base_url: str, user_agent: str = USER_AGENT):
        """
        Initialize robots.txt parser
        
        Args:
            base_url: Base URL of the website
            user_agent: User agent string
        """
        self.base_url = base_url.rstrip('/')
        self.user_agent = user_agent
        self.robots_url = f"{self.base_url}/robots.txt"
        self.parser = RobotFileParser()
        self.parser.set_url(self.robots_url)
        self.content = None
        self.sitemap_urls = []
    
    def fetch_and_parse(self) -> bool:
        """
        Fetch and parse robots.txt
        
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(self.robots_url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                self.content = response.text
                
                # Parse with RobotFileParser
                self.parser.parse(self.content.split('\n'))
                
                # Extract sitemap URLs
                for line in self.content.split('\n'):
                    if line.lower().startswith('sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        self.sitemap_urls.append(sitemap_url)
                
                logger.info(f"Successfully parsed robots.txt from {self.robots_url}")
                logger.info(f"Found {len(self.sitemap_urls)} sitemap(s)")
                return True
            else:
                logger.warning(f"robots.txt not found at {self.robots_url} (status: {response.status_code})")
                return False
        
        except Exception as e:
            logger.error(f"Error fetching robots.txt: {e}")
            return False
    
    def can_fetch(self, url: str) -> bool:
        """
        Check if URL can be fetched according to robots.txt
        
        Args:
            url: URL to check
        
        Returns:
            True if allowed, False if disallowed
        """
        return self.parser.can_fetch(self.user_agent, url)
    
    def get_crawl_delay(self) -> Optional[float]:
        """
        Get crawl delay from robots.txt
        
        Returns:
            Crawl delay in seconds or None
        """
        delay = self.parser.crawl_delay(self.user_agent)
        return delay
    
    def get_sitemap_urls(self) -> List[str]:
        """Get list of sitemap URLs from robots.txt"""
        return self.sitemap_urls
    
    def print_info(self):
        """Print robots.txt information - feedback function"""
        print("\n" + "="*60)
        print("ROBOTS.TXT INFORMATION")
        print("="*60)
        print(f"URL: {self.robots_url}")
        print(f"User Agent: {self.user_agent}")
        
        delay = self.get_crawl_delay()
        if delay:
            print(f"Crawl Delay: {delay} seconds")
        else:
            print("Crawl Delay: Not specified")
        
        print(f"\nSitemaps found: {len(self.sitemap_urls)}")
        for sitemap in self.sitemap_urls:
            print(f"  - {sitemap}")
        
        print("\n" + "="*60 + "\n")


class SitemapBasedScraper:
    """
    Helper class to scrape websites using sitemap
    """
    
    def __init__(self, base_url: str, user_agent: str = USER_AGENT):
        """
        Initialize sitemap-based scraper
        
        Args:
            base_url: Base URL of website
            user_agent: User agent string
        """
        self.base_url = base_url
        self.robots = RobotsParser(base_url, user_agent)
        self.sitemap = SitemapParser(base_url)
    
    def get_allowed_urls(self, filter_patterns: Optional[List[str]] = None) -> List[Dict[str, any]]:
        """
        Get all URLs that are allowed by robots.txt
        
        Args:
            filter_patterns: Optional regex patterns to filter URLs
        
        Returns:
            List of allowed URL dictionaries
        """
        # Parse robots.txt
        self.robots.fetch_and_parse()
        
        # Get all URLs from sitemap
        all_urls = self.sitemap.get_all_urls(self.robots.content)
        
        # Filter by robots.txt rules
        allowed_urls = []
        for url_data in all_urls:
            url = url_data['url']
            
            if self.robots.can_fetch(url):
                # Apply additional filters if provided
                if filter_patterns:
                    from utils.validation import matches_pattern
                    if matches_pattern(url, filter_patterns):
                        allowed_urls.append(url_data)
                else:
                    allowed_urls.append(url_data)
            else:
                logger.debug(f"URL disallowed by robots.txt: {url}")
        
        logger.info(f"Total URLs from sitemap: {len(all_urls)}")
        logger.info(f"Allowed URLs: {len(allowed_urls)}")
        
        return allowed_urls
    
    def print_summary(self, urls: List[Dict[str, any]]):
        """Print summary of sitemap URLs - feedback function"""
        print("\n" + "="*60)
        print("SITEMAP SCRAPING SUMMARY")
        print("="*60)
        print(f"Base URL: {self.base_url}")
        print(f"Total URLs: {len(urls)}")
        
        if urls:
            # Count by priority
            priorities = {}
            for url_data in urls:
                priority = url_data.get('priority', 'unknown')
                priorities[priority] = priorities.get(priority, 0) + 1
            
            print("\nBy Priority:")
            for priority, count in sorted(priorities.items(), reverse=True):
                print(f"  {priority}: {count} URLs")
            
            # Count by change frequency
            changefreqs = {}
            for url_data in urls:
                changefreq = url_data.get('changefreq', 'unknown')
                changefreqs[changefreq] = changefreqs.get(changefreq, 0) + 1
            
            print("\nBy Change Frequency:")
            for freq, count in changefreqs.items():
                print(f"  {freq}: {count} URLs")
            
            # Sample URLs
            print("\nSample URLs:")
            for url_data in urls[:5]:
                print(f"  - {url_data['url']}")
                if 'priority' in url_data:
                    print(f"    Priority: {url_data['priority']}")
                if 'lastmod' in url_data:
                    print(f"    Last Modified: {url_data['lastmod']}")
        
        print("\n" + "="*60 + "\n")


def check_robots_and_sitemap(base_url: str) -> Dict[str, any]:
    """
    Check robots.txt and sitemap for a website - feedback function
    
    Args:
        base_url: Base URL of website
    
    Returns:
        Dictionary with robots and sitemap info
    """
    print(f"\nAnalyzing {base_url}...")
    
    scraper = SitemapBasedScraper(base_url)
    
    # Check robots.txt
    robots_success = scraper.robots.fetch_and_parse()
    scraper.robots.print_info()
    
    # Get sitemap URLs
    urls = scraper.get_allowed_urls()
    scraper.print_summary(urls)
    
    # Get crawl delay recommendation
    crawl_delay = scraper.robots.get_crawl_delay()
    
    return {
        'base_url': base_url,
        'robots_exists': robots_success,
        'sitemap_urls': scraper.robots.get_sitemap_urls(),
        'total_urls': len(urls),
        'allowed_urls': urls,
        'recommended_crawl_delay': crawl_delay
    }


# Integration functions for existing scraper

def prepare_url_queue_from_sitemap(base_url: str, 
                                    include_patterns: Optional[List[str]] = None,
                                    exclude_patterns: Optional[List[str]] = None) -> List[str]:
    """
    Prepare a queue of URLs from sitemap for scraping
    
    Args:
        base_url: Base URL of website
        include_patterns: Patterns to include
        exclude_patterns: Patterns to exclude
    
    Returns:
        List of URLs to scrape
    """
    scraper = SitemapBasedScraper(base_url)
    
    # Get allowed URLs
    url_dicts = scraper.get_allowed_urls()
    
    # Filter by patterns
    from utils.validation import should_scrape_url
    
    filtered_urls = []
    for url_data in url_dicts:
        url = url_data['url']
        if should_scrape_url(url, include_patterns or [], exclude_patterns or []):
            filtered_urls.append(url)
    
    logger.info(f"Prepared {len(filtered_urls)} URLs from sitemap for scraping")
    return filtered_urls


def get_recommended_rate_limit(base_url: str, default: float = 1.5) -> float:
    """
    Get recommended rate limit from robots.txt
    
    Args:
        base_url: Base URL of website
        default: Default rate limit if not specified
    
    Returns:
        Recommended rate limit in seconds
    """
    robots = RobotsParser(base_url)
    robots.fetch_and_parse()
    
    crawl_delay = robots.get_crawl_delay()
    
    if crawl_delay:
        logger.info(f"Using crawl delay from robots.txt: {crawl_delay}s")
        return crawl_delay
    else:
        logger.info(f"No crawl delay specified, using default: {default}s")
        return default
