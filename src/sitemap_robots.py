"""
Parse sitemap.xml and robots.txt from websites
"""
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
from typing import List, Optional, Tuple
from urllib.robotparser import RobotFileParser


class SitemapRobotsParser:
    """Parse sitemap and robots.txt for a website"""
    
    def __init__(self, base_url: str, user_agent: str = "ScrapeBot"):
        self.base_url = base_url.rstrip('/')
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
    
    def get_robots_txt(self) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Fetch and parse robots.txt
        
        Returns:
            (success, content, crawl_delay)
        """
        robots_url = f"{self.base_url}/robots.txt"
        
        try:
            response = self.session.get(robots_url, timeout=10)
            if response.status_code == 200:
                content = response.text
                
                # Parse with RobotFileParser
                parser = RobotFileParser()
                parser.parse(content.split('\n'))
                
                # Get crawl delay
                crawl_delay = parser.crawl_delay(self.user_agent)
                
                return True, content, crawl_delay
            else:
                return False, None, None
        except Exception as e:
            print(f"Error fetching robots.txt: {e}")
            return False, None, None
    
    def find_sitemap_url(self, robots_content: Optional[str] = None) -> Optional[str]:
        """
        Find sitemap URL from robots.txt or common locations
        
        Returns:
            Sitemap URL or None
        """
        # Try to get from robots.txt
        if robots_content:
            for line in robots_content.split('\n'):
                if line.lower().startswith('sitemap:'):
                    sitemap_url = line.split(':', 1)[1].strip()
                    return sitemap_url
        
        # Try common locations
        common_paths = [
            '/sitemap.xml',
            '/sitemap_index.xml',
            '/sitemap1.xml',
        ]
        
        for path in common_paths:
            sitemap_url = self.base_url + path
            try:
                response = self.session.head(sitemap_url, timeout=10)
                if response.status_code == 200:
                    return sitemap_url
            except:
                continue
        
        return None
    
    def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Parse sitemap XML and extract all URLs
        
        Returns:
            List of URLs
        """
        urls = []
        
        try:
            response = self.session.get(sitemap_url, timeout=10)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            # Handle namespace
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
            # Check if sitemap index or regular sitemap
            if root.tag.endswith('sitemapindex'):
                # It's an index, get all nested sitemaps
                for sitemap in root.findall('ns:sitemap', namespace):
                    loc = sitemap.find('ns:loc', namespace)
                    if loc is not None and loc.text:
                        # Recursively parse nested sitemap
                        nested_urls = self.parse_sitemap(loc.text)
                        urls.extend(nested_urls)
            else:
                # Regular sitemap with URLs
                for url_elem in root.findall('ns:url', namespace):
                    loc = url_elem.find('ns:loc', namespace)
                    if loc is not None and loc.text:
                        urls.append(loc.text)
        
        except Exception as e:
            print(f"Error parsing sitemap: {e}")
        
        return urls
    
    def get_all_urls(self) -> Tuple[List[str], Optional[float]]:
        """
        Get all URLs from sitemap and recommended crawl delay
        
        Returns:
            (list of URLs, crawl_delay)
        """
        # Get robots.txt
        success, robots_content, crawl_delay = self.get_robots_txt()
        
        # Find sitemap
        sitemap_url = self.find_sitemap_url(robots_content)
        
        if not sitemap_url:
            print("No sitemap found")
            return [], crawl_delay
        
        print(f"Found sitemap: {sitemap_url}")
        
        # Parse sitemap
        urls = self.parse_sitemap(sitemap_url)
        
        print(f"Extracted {len(urls)} URLs from sitemap")
        
        return urls, crawl_delay
