"""
Static scraper for websites that don't require JavaScript
Uses requests + BeautifulSoup
"""
import requests
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
from bs4 import BeautifulSoup
from collections import deque

from config.settings import USER_AGENT, REQUEST_TIMEOUT, RETRY_ATTEMPTS, RETRY_DELAY
from scrapers.base_scraper import BaseScraper
from utils.logger import log_scrape_start, log_scrape_success, log_scrape_error
from utils.database import DatabaseManager


class StaticScraper(BaseScraper):
    """
    Scraper for static HTML websites
    Uses HTTP requests and BeautifulSoup for parsing
    """
    
    def __init__(self, config: Dict[str, Any], db_path: str):
        """
        Initialize static scraper
        
        Args:
            config: Scraper configuration
            db_path: Database path
        """
        super().__init__(config, db_path)
        
        # Setup requests session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def scrape_page(self, url: str, retry_count: int = 0) -> Optional[Dict[str, Any]]:
        """
        Scrape a single page
        
        Args:
            url: URL to scrape
            retry_count: Current retry attempt
        
        Returns:
            Dictionary with page data or None if failed
        """
        log_scrape_start(url)
        
        try:
            # Respect rate limiting
            self._respect_rate_limit()
            
            # Make request
            start_time = time.time()
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response_time = time.time() - start_time
            
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract data
            page_data = self._extract_page_data(soup, url, response, response_time)
            
            # Validate
            if not self.validate_scraped_data(page_data):
                self.logger.warning(f"Data validation failed for {url}")
                return None
            
            log_scrape_success(url, response.status_code, len(response.content), response_time)
            
            return page_data
            
        except requests.RequestException as e:
            log_scrape_error(url, e, retry_count)
            
            # Retry if not exceeded max attempts
            if retry_count < RETRY_ATTEMPTS:
                self.logger.info(f"Retrying {url} (attempt {retry_count + 1}/{RETRY_ATTEMPTS})")
                time.sleep(RETRY_DELAY * (retry_count + 1))  # Exponential backoff
                return self.scrape_page(url, retry_count + 1)
            
            return None
            
        except Exception as e:
            log_scrape_error(url, e, retry_count)
            return None
    
    def _extract_page_data(self, soup: BeautifulSoup, url: str, 
                          response: requests.Response, response_time: float) -> Dict[str, Any]:
        """
        Extract all relevant data from the page
        
        Args:
            soup: BeautifulSoup object
            url: Page URL
            response: HTTP response object
            response_time: Time taken for request
        
        Returns:
            Dictionary with extracted data
        """
        # Basic metadata
        title = self._extract_title(soup)
        description = self._extract_description(soup)
        
        # Main content
        main_content = self._extract_main_content(soup)
        text_content = main_content.get_text(separator=' ', strip=True) if main_content else ''
        
        # Structured content
        headings = self._extract_headings(soup)
        paragraphs = self._extract_paragraphs(soup)
        
        # Links and media
        links = self._extract_links(soup, url)
        images = self._extract_images(soup, url)
        videos = self._extract_videos(soup, url)
        
        # Additional content
        tables = self._extract_tables(soup)
        code_blocks = self._extract_code_blocks(soup)
        
        # Page classification
        page_type = self._classify_page(url)
        
        # Compile page data
        page_data = {
            # URLs and identification
            'url': url,
            'parent_url': None,  # To be set by caller
            'depth': 0,  # To be set by caller
            
            # Metadata
            'title': title,
            'description': description,
            'page_type': page_type,
            'language': self._detect_language(soup),
            
            # Content
            'text_content': text_content[:10000],  # Limit for database
            'full_content': {
                'headings': headings,
                'paragraphs': paragraphs,
                'tables': tables,
                'code_blocks': code_blocks,
            },
            
            # Links and media
            'links': links,
            'links_count': len(links),
            'images': images,
            'images_count': len(images),
            'videos': videos,
            'videos_count': len(videos),
            
            # Technical metadata
            'status_code': response.status_code,
            'content_type': response.headers.get('content-type', ''),
            'content_length': len(response.content),
            'response_time': response_time,
            'last_modified': response.headers.get('last-modified'),
            
            # Timestamps
            'scraped_at': datetime.utcnow().isoformat(),
            
            # Status
            'is_successful': True,
            'has_errors': False,
            'error_message': None,
        }
        
        return page_data
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title"""
        title_selector = self.selectors.get('title', 'h1, title')
        
        # Try h1 first
        h1 = soup.select_one('h1')
        if h1:
            return h1.get_text(strip=True)
        
        # Try title tag
        title = soup.find('title')
        if title:
            return title.get_text(strip=True)
        
        return None
    
    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page description"""
        desc_selector = self.selectors.get('description', 'meta[name="description"]')
        
        meta_desc = soup.select_one(desc_selector)
        if meta_desc:
            return meta_desc.get('content', '').strip()
        
        return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Extract main content area"""
        content_selector = self.selectors.get('main_content', 'main, article, .content, #content')
        
        main = soup.select_one(content_selector)
        if main:
            return main
        
        # Fallback to body
        return soup.find('body')
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract all headings with hierarchy"""
        headings = []
        
        for tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            for heading in soup.find_all(tag_name):
                headings.append({
                    'level': int(tag_name[1]),
                    'text': heading.get_text(strip=True)
                })
        
        return headings
    
    def _extract_paragraphs(self, soup: BeautifulSoup) -> List[str]:
        """Extract all paragraph text"""
        paragraphs = []
        
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if text and len(text) > 20:  # Filter out very short paragraphs
                paragraphs.append(text)
        
        return paragraphs
    
    def _extract_videos(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Extract video elements and embedded videos"""
        videos = []
        video_selector = self.selectors.get('videos', 'video[src], iframe[src*="youtube"], iframe[src*="vimeo"]')
        
        for video in soup.select(video_selector):
            src = video.get('src', '').strip()
            if not src:
                continue
            
            videos.append({
                'url': src,
                'type': 'iframe' if video.name == 'iframe' else 'video'
            })
        
        return videos
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[List[List[str]]]:
        """Extract table data"""
        tables = []
        
        for table in soup.find_all('table'):
            table_data = []
            for row in table.find_all('tr'):
                row_data = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                if row_data:
                    table_data.append(row_data)
            
            if table_data:
                tables.append(table_data)
        
        return tables
    
    def _extract_code_blocks(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract code blocks"""
        code_blocks = []
        
        for pre in soup.find_all('pre'):
            code = pre.find('code')
            if code:
                code_blocks.append({
                    'language': code.get('class', [''])[0].replace('language-', ''),
                    'code': code.get_text()
                })
            else:
                code_blocks.append({
                    'language': 'text',
                    'code': pre.get_text()
                })
        
        return code_blocks
    
    def _detect_language(self, soup: BeautifulSoup) -> Optional[str]:
        """Detect page language"""
        html = soup.find('html')
        if html:
            return html.get('lang')
        return None
    
    def scrape_website(self, start_url: Optional[str] = None, 
                      max_pages: Optional[int] = None,
                      use_sitemap: Optional[bool] = None) -> Dict[str, Any]:
        """
        Scrape entire website using sitemap or BFS
        
        Args:
            start_url: Starting URL (defaults to base_url)
            max_pages: Maximum number of pages to scrape
            use_sitemap: Whether to use sitemap (overrides config)
        
        Returns:
            Dictionary with scraping results
        """
        start_url = start_url or self.base_url
        self.stats['start_time'] = datetime.utcnow()
        
        # Override config if specified
        if use_sitemap is not None:
            self.use_sitemap = use_sitemap
        
        self.logger.info(f"Starting scrape of {self.name} from {start_url}")
        
        # Initialize database
        with DatabaseManager(self.db_path) as db:
            # Get or create website
            website, created = db.get_or_create_website(
                domain=self.domain,
                name=self.name,
                base_url=self.base_url,
                scraper_type='static'
            )
            
            # Create scrape session
            scrape_session_id = db.create_scrape_session(
                website_id=website['id'],
                config=self.config
            )
        
        # Try to get URLs from sitemap first
        sitemap_urls = self.check_robots_and_get_sitemap() if self.use_sitemap else None
        
        if sitemap_urls:
            # Scrape from sitemap
            self.logger.info(f"Using sitemap with {len(sitemap_urls)} URLs")
            result = self._scrape_from_sitemap(
                sitemap_urls, website['id'], scrape_session_id, max_pages
            )
        else:
            # Fall back to traditional crawling
            self.logger.info("Using traditional BFS crawling")
            result = self._scrape_with_bfs(
                start_url, website['id'], scrape_session_id, max_pages
            )
        
        # Complete session
        self.stats['end_time'] = datetime.utcnow()
        
        with DatabaseManager(self.db_path) as db:
            db.complete_scrape_session(
                scrape_session_id,
                status='completed'
            )
            db.update_website_stats(website['id'])
        
        self.print_summary()
        
        return {
            'website_id': website['id'],
            'session_id': scrape_session_id,
            'stats': self.stats
        }
    
    def _scrape_from_sitemap(self, urls: List[str], website_id: int,
                            session_id: int, max_pages: Optional[int]) -> Dict[str, Any]:
        """
        Scrape pages from sitemap URL list
        
        Args:
            urls: List of URLs from sitemap
            website_id: Website ID
            session_id: Session ID
            max_pages: Maximum pages to scrape
        
        Returns:
            Scraping results
        """
        self.logger.info(f"Scraping {len(urls)} URLs from sitemap")
        
        try:
            for i, url in enumerate(urls):
                if max_pages and self.stats['pages_scraped'] >= max_pages:
                    self.logger.info(f"Reached max_pages limit: {max_pages}")
                    break
                
                # Check if should scrape
                if not self._should_scrape_url(url):
                    continue
                
                # Mark as visited
                self.visited_urls.add(url)
                
                # Scrape page
                page_data = self.scrape_page(url)
                
                if page_data:
                    # Set depth to 0 for sitemap URLs
                    page_data['depth'] = 0
                    page_data['parent_url'] = None
                    
                    # Save page
                    page_id = self.save_page(page_data, website_id, session_id)
                    
                    if page_id:
                        # Save links to database
                        with DatabaseManager(self.db_path) as db:
                            for link in page_data['links']:
                                db.create_link(
                                    source_page_id=page_id,
                                    url=link['url'],
                                    anchor_text=link['anchor_text'],
                                    link_type=link['link_type']
                                )
                        
                        self.stats['pages_scraped'] += 1
                        self.stats['total_bytes'] += page_data['content_length']
                        self.print_progress()
                else:
                    self.stats['pages_failed'] += 1
                    self.failed_urls.add(url)
        
        except KeyboardInterrupt:
            self.logger.warning("Scraping interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error during sitemap scraping: {e}")
        
        return {'method': 'sitemap'}
    
    def _scrape_with_bfs(self, start_url: str, website_id: int,
                        session_id: int, max_pages: Optional[int]) -> Dict[str, Any]:
        """
        Traditional BFS crawling (original method)
        
        Args:
            start_url: Starting URL
            website_id: Website ID
            session_id: Session ID  
            max_pages: Maximum pages to scrape
        
        Returns:
            Scraping results
        """
        # BFS queue: (url, depth, parent_url)
        queue = deque([(start_url, 0, None)])
        
        try:
            while queue and (max_pages is None or self.stats['pages_scraped'] < max_pages):
                url, depth, parent_url = queue.popleft()
                
                # Check if should scrape
                if not self._should_scrape_url(url):
                    continue
                
                # Check depth limit
                if depth > self.max_depth:
                    continue
                
                # Mark as visited
                self.visited_urls.add(url)
                
                # Scrape page
                page_data = self.scrape_page(url)
                
                if page_data:
                    # Set depth and parent
                    page_data['depth'] = depth
                    page_data['parent_url'] = parent_url
                    
                    # Save page
                    page_id = self.save_page(page_data, website.id, scrape_session.id)
                    
                    if page_id:
                        # Save links to database
                        with DatabaseManager(self.db_path) as db:
                            for link in page_data['links']:
                                if link['link_type'] == 'internal':
                                    db.create_link(
                                        source_page_id=page_id,
                                        url=link['url'],
                                        anchor_text=link['anchor_text'],
                                        link_type=link['link_type']
                                    )
                                    
                                    # Add to queue if not visited
                                    if link['url'] not in self.visited_urls:
                                        queue.append((link['url'], depth + 1, url))
                        
                        self.stats['pages_scraped'] += 1
                        self.stats['total_bytes'] += page_data['content_length']
                        self.print_progress()
                else:
                    self.stats['pages_failed'] += 1
                    self.failed_urls.add(url)
        
        except KeyboardInterrupt:
            self.logger.warning("Scraping interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Unexpected error during BFS scraping: {e}")
        
        return {'method': 'bfs'}

