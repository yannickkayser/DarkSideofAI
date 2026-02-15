"""
Main scraper for JavaScript-heavy websites using Playwright
"""
import time
import json
import re
import hashlib
from datetime import datetime
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Optional
from playwright.sync_api import sync_playwright, Page, Browser
from bs4 import BeautifulSoup

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    HEADLESS, PAGE_WAIT_TIMEOUT, NETWORK_IDLE_TIMEOUT,
    RATE_LIMIT_DELAY, RETRY_ATTEMPTS, RETRY_DELAY, RAW_DATA_DIR 
)
from database import Database
from sitemap_robots import SitemapRobotsParser
from validator import validate_page_data
from logger import (
    setup_logger, log_scrape_start, log_scrape_success, 
    log_scrape_error, log_validation_result
)


class WebScraper:
    """Scraper for JavaScript-heavy websites"""
    
    def __init__(self, website_config: dict, db: Database):
        self.config = website_config
        self.db = db
        self.base_url = website_config['base_url']
        self.domain = urlparse(self.base_url).netloc
        self.rate_limit = website_config.get('rate_limit', RATE_LIMIT_DELAY)
        self.max_depth = website_config.get('max_depth', 3)
        
        # Setup logger
        self.logger = setup_logger(
            f"scraper.{self.domain}",
            f"logs/scraper_{self.domain}.log"
        )
        
        # Tracking
        self.visited_urls = set()
        self.last_request_time = 0
        
        # Stats
        self.stats = {
            'pages_scraped': 0,
            'pages_failed': 0,
            'total_bytes': 0
        }
    
    def _wait_for_rate_limit(self):
        """Respect rate limiting"""
        if self.last_request_time > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit:
                sleep_time = self.rate_limit - elapsed
                time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _extract_directory(self, url: str) -> str:
        """Extract directory path from URL"""
        parsed = urlparse(url)
        path = parsed.path.rstrip('/')
        if not path or path == '/':
            return '/'
        return '/'.join(path.split('/')[:-1]) or '/'
    
    def _extract_css_colors(self, page: Page, soup) -> dict:
        """
        Extract color codes categorized by element type
        
        Returns:
            Dictionary with color lists by category
        """
        colors = {
            'background_colors': [],
            'text_colors': [],
            'link_colors': [],
            'button_colors': []
        }
        
        try:
            # Simpler approach: evaluate each color type separately
            
            # Background colors
            bg_script = """
            Array.from(new Set(
                Array.from(document.querySelectorAll('body, header, main, section, nav, footer'))
                    .map(el => window.getComputedStyle(el).backgroundColor)
                    .filter(c => c && c !== 'rgba(0, 0, 0, 0)')
            )).slice(0, 20)
            """
            colors['background_colors'] = page.evaluate(bg_script)
            
            # Text colors
            text_script = """
            Array.from(new Set(
                Array.from(document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, span'))
                    .map(el => window.getComputedStyle(el).color)
                    .filter(c => c)
            )).slice(0, 20)
            """
            colors['text_colors'] = page.evaluate(text_script)
            
            # Link colors
            link_script = """
            Array.from(new Set(
                Array.from(document.querySelectorAll('a'))
                    .map(el => window.getComputedStyle(el).color)
                    .filter(c => c)
            )).slice(0, 10)
            """
            colors['link_colors'] = page.evaluate(link_script)
            
            # Button colors
            button_script = """
            Array.from(new Set(
                Array.from(document.querySelectorAll('button, .btn, [role="button"]'))
                    .flatMap(el => [
                        window.getComputedStyle(el).backgroundColor,
                        window.getComputedStyle(el).color
                    ])
                    .filter(c => c && c !== 'rgba(0, 0, 0, 0)')
            )).slice(0, 10)
            """
            colors['button_colors'] = page.evaluate(button_script)
            
        except Exception as e:
            self.logger.warning(f"Error extracting CSS colors: {e}")
            # Return empty lists on error
            colors = {
                'background_colors': [],
                'text_colors': [],
                'link_colors': [],
                'button_colors': []
            }
        
        return colors
    
    def _extract_text_from_page(self, page: Page) -> Dict[str, any]:
        """
        Extract all text content and metadata from page
        
        Returns:
            Dictionary with page data
        """
        # Wait for page to be fully loaded
        try:
            page.wait_for_load_state('networkidle', timeout=NETWORK_IDLE_TIMEOUT)
        except:
            pass  # Continue even if timeout
        
        # Get page content
        html_content = page.content()
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract CSS colors 
        css_colors = self._extract_css_colors(page, soup)

        # Extract title
        title_elem = soup.find('title')
        title = title_elem.get_text(strip=True) if title_elem else ''
        
        if not title:
            h1 = soup.find('h1')
            title = h1.get_text(strip=True) if h1 else 'Untitled'
        
        # Extract main content areas
        main_selectors = ['main', 'article', '[role="main"]', '.content', '#content']
        main_content = None
        
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body')
        
        # Extract text from main content
        if main_content:
            # Remove script and style elements
            for script in main_content(['script', 'style', 'noscript']):
                script.decompose()
            
            text_content = main_content.get_text(separator=' ', strip=True)
            html_element = main_content.name
        else:
            text_content = ''
            html_element = 'body'
        
        # Extract all links
        links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href', '').strip()
            if href and not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                full_url = urljoin(page.url, href)
                parsed_link = urlparse(full_url)
                parsed_base = urlparse(self.base_url)
                
                link_type = 'internal' if parsed_link.netloc == parsed_base.netloc else 'external'
                
                links.append({
                    'url': full_url,
                    'anchor_text': link.get_text(strip=True),
                    'type': link_type
                })
        
        return {
            'url': page.url,
            'title': title,
            'text_content': text_content,
            'html_element': html_element,
            'directory': self._extract_directory(page.url),
            'content_length': len(html_content),
            'css_colors': css_colors,
            'links': links
        }
    
    def _save_raw_data(self, page_data: dict):
        """Save raw page data as JSON backup"""
        # Create domain-specific directory
        domain_dir = RAW_DATA_DIR / self.domain.replace('.', '_')
        domain_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        url_hash = hashlib.sha256(page_data['url'].encode()).hexdigest()[:16]
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"{url_hash}_{timestamp}.json"
        
        file_path = domain_dir / filename
        
        # Save to JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"Saved raw data to {file_path}")
    
    def scrape_page(self, page: Page, url: str, depth: int = 0) -> Optional[Dict]:
        """Scrape a single page"""
        log_scrape_start(self.logger, url)
        
        for attempt in range(RETRY_ATTEMPTS):
            try:
                self._wait_for_rate_limit()
                
                # ADD: Check if page is valid
                if page.is_closed():
                    self.logger.error("Page is closed, cannot scrape")
                    return None
                
                # Navigate to page
                response = page.goto(url, wait_until='domcontentloaded', timeout=PAGE_WAIT_TIMEOUT)
                
                if not response:
                    raise Exception("No response from page")
                
                status_code = response.status
                
                # Extract content
                page_data = self._extract_text_from_page(page)
                page_data['status_code'] = status_code
                page_data['depth'] = depth
                
                # Validate
                is_valid, message = validate_page_data(page_data)
                log_validation_result(self.logger, is_valid, f"{url}: {message}")
                
                if not is_valid:
                    self.logger.warning(f"Validation failed for {url}: {message}")
                    return None
                
                log_scrape_success(self.logger, url, status_code, page_data['content_length'])
                
                return page_data
                
            except KeyboardInterrupt:  # ADD THIS
                raise  # Re-raise to stop scraping
            except Exception as e:
                error_msg = str(e)
                
                # Don't retry if page/browser closed errors
                if 'closed' in error_msg.lower() or 'target' in error_msg.lower():
                    log_scrape_error(self.logger, url, error_msg)
                    return None
                
                log_scrape_error(self.logger, url, error_msg)
                
                if attempt < RETRY_ATTEMPTS - 1:
                    self.logger.info(f"Retrying {url} (attempt {attempt + 2}/{RETRY_ATTEMPTS})")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return None
        
        return None
    
    def scrape_website(self, max_pages: Optional[int] = None) -> Dict:
        """
        Scrape entire website using sitemap
        """
        self.logger.info(f"Starting scrape of {self.config['name']}")
        
        # Get sitemap URLs
        parser = SitemapRobotsParser(self.base_url)
        sitemap_urls, crawl_delay = parser.get_all_urls()
        
        if crawl_delay and crawl_delay > self.rate_limit:
            self.logger.info(f"Using crawl delay from robots.txt: {crawl_delay}s")
            self.rate_limit = crawl_delay
        
        if not sitemap_urls:
            self.logger.warning("No sitemap found, scraping only base URL")
            sitemap_urls = [self.base_url]
        
        # Add website to database
        website_id = self.db.add_website(
            domain=self.domain,
            name=self.config['name'],
            base_url=self.base_url,
            website_type=self.config.get('type', 'Unknown')
        )
        
        ## Start Playwright
        with sync_playwright() as p:
            # More realistic browser args
            browser = p.chromium.launch(
                headless=HEADLESS,
                args=[
                    '--disable-blink-features=AutomationControlled',  
                    '--disable-dev-shm-usage',
                    '--no-sandbox'
                ]
            )
            
            #  More realistic context
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York',
                
                extra_http_headers={
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
            )
            
            # Stealth settings
            context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)
            
            
            context.set_default_timeout(PAGE_WAIT_TIMEOUT)
            
            page = context.new_page()
            
            try:
                # Scrape pages from sitemap
                for i, url in enumerate(sitemap_urls):
                    if max_pages and self.stats['pages_scraped'] >= max_pages:
                        self.logger.info(f"Reached max_pages limit: {max_pages}")
                        break
                    
                    # Skip if already visited
                    if url in self.visited_urls or self.db.page_exists(url):
                        continue
                    
                    self.visited_urls.add(url)
                    
                    # ADD: Check if page is still valid, recreate if needed
                    try:
                        if page.is_closed():
                            page = context.new_page()
                    except:
                        page = context.new_page()
                    
                    # Scrape page
                    page_data = self.scrape_page(page, url, depth=0)
                    
                    if page_data:
                        # Save raw data backup
                        self._save_raw_data(page_data)
                        
                        # Save to database
                        page_id = self.db.add_page(
                            website_id=website_id,
                            url=page_data['url'],
                            title=page_data['title'],
                            text_content=page_data['text_content'],
                            directory=page_data['directory'],
                            html_element=page_data['html_element'],
                            depth=page_data['depth'],
                            status_code=page_data['status_code'],
                            content_length=page_data['content_length'],
                            css_colors=json.dumps(page_data['css_colors'])
                        )
                        
                        if page_id:
                            # Save links
                            for link in page_data['links']:
                                self.db.add_link(
                                    source_page_id=page_id,
                                    target_url=link['url'],
                                    anchor_text=link['anchor_text'],
                                    link_type=link['type']
                                )
                            
                            self.stats['pages_scraped'] += 1
                            self.stats['total_bytes'] += page_data['content_length']
                        
                        print(f"\rProgress: {self.stats['pages_scraped']} pages scraped", end='')
                    else:
                        self.stats['pages_failed'] += 1
            
            except KeyboardInterrupt:  # ADD THIS
                self.logger.warning("Scraping interrupted by user")
            except Exception as e:  # ADD THIS
                self.logger.error(f"Unexpected error: {e}")
            finally:
                try:
                    if not page.is_closed():
                        page.close()
                    context.close()
                    browser.close()
                except:
                    pass  # Ignore close errors
        
        # Update website
        self.db.update_website_last_scraped(website_id)
        
        # Print summary
        self._print_summary()
        
        return self.stats
    
    def _print_summary(self):
        """Print scraping summary"""
        print("\n\n" + "="*60)
        print(f"SCRAPING SUMMARY - {self.config['name']}")
        print("="*60)
        print(f"Pages Scraped: {self.stats['pages_scraped']}")
        print(f"Pages Failed: {self.stats['pages_failed']}")
        print(f"Total Data: {self.stats['total_bytes'] / (1024*1024):.2f} MB")
        print("="*60 + "\n")
