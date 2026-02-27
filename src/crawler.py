"""
BFS link crawler for websites without a sitemap.
Discovers URLs by following internal links, then hands them to WebScraper.
"""

import time
from collections import deque
from urllib.parse import urlparse, urljoin, urldefrag
from typing import Set, List, Optional

import requests
from bs4 import BeautifulSoup


class LinkCrawler:
    """
    Crawls a website by following internal links (BFS).
    Use this for sites that have no sitemap (e.g. crowdgen.com, telusinternational.ai).

    Usage:
        crawler = LinkCrawler("https://crowdgen.com", max_pages=200, max_depth=3)
        urls = crawler.discover_urls()
        # Pass urls to WebScraper or use standalone
    """

    def __init__(
        self,
        base_url: str,
        max_pages: int = 500,
        max_depth: int = 3,
        rate_limit: float = 1.5,
        user_agent: str = "Mozilla/5.0 (compatible; ScrapeBot/1.0)",
        exclude_extensions: Optional[Set[str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.rate_limit = rate_limit

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        self.exclude_extensions = exclude_extensions or {
            ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
            ".zip", ".tar", ".gz", ".mp4", ".mp3", ".avi", ".css", ".js",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_internal(self, url: str) -> bool:
        return urlparse(url).netloc == self.domain

    def _normalize(self, url: str) -> Optional[str]:
        """Remove fragments and trailing slashes; return None if excluded."""
        url, _ = urldefrag(url)
        url = url.rstrip("/") or "/"
        path = urlparse(url).path.lower()
        if any(path.endswith(ext) for ext in self.exclude_extensions):
            return None
        return url

    def _fetch_links(self, url: str) -> List[str]:
        """Fetch a page with requests and extract internal hrefs."""
        try:
            resp = self.session.get(url, timeout=10, allow_redirects=True)
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, "html.parser")
            links = []
            for tag in soup.find_all("a", href=True):
                href = urljoin(url, tag["href"])
                norm = self._normalize(href)
                if norm and self._is_internal(norm):
                    links.append(norm)
            return links
        except Exception as e:
            print(f"  [crawler] Error fetching {url}: {e}")
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_urls(self) -> List[str]:
        """
        BFS crawl starting from base_url.

        Returns:
            Ordered list of discovered internal URLs (including base_url).
        """
        visited: Set[str] = set()
        queue: deque = deque()  # (url, depth)

        start = self._normalize(self.base_url) or self.base_url
        queue.append((start, 0))
        visited.add(start)

        print(f"[crawler] Starting BFS on {self.base_url}")
        print(f"[crawler] Limits — max_pages={self.max_pages}, max_depth={self.max_depth}")

        while queue and len(visited) < self.max_pages:
            url, depth = queue.popleft()

            print(f"[crawler] ({len(visited)}/{self.max_pages}) depth={depth} {url}")

            if depth < self.max_depth:
                links = self._fetch_links(url)
                for link in links:
                    if link not in visited and len(visited) < self.max_pages:
                        visited.add(link)
                        queue.append((link, depth + 1))

            time.sleep(self.rate_limit)

        discovered = list(visited)
        print(f"[crawler] Done — {len(discovered)} URLs discovered.")
        return discovered


# ----------------------------------------------------------------------
# Standalone entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.config import get_website_config, list_websites, DB_PATH
    from database import Database
    from scraper import WebScraper

    def crawl_and_scrape(domain: str, max_pages: int = 100, max_depth: int = 3):
        config = get_website_config(domain)
        if not config:
            print(f"Error: no config for '{domain}'. Available: {', '.join(list_websites())}")
            return

        # Discover URLs via BFS
        crawler = LinkCrawler(
            base_url=config["base_url"],
            max_pages=max_pages,
            max_depth=max_depth,
            rate_limit=config.get("rate_limit", 1.5),
        )
        urls = crawler.discover_urls()

        if not urls:
            print("No URLs discovered — aborting.")
            return

        # Inject discovered URLs into WebScraper by monkey-patching sitemap step
        db = Database(str(DB_PATH))
        scraper = WebScraper(config, db)

        # Override sitemap discovery so scraper uses our crawled URLs
        from sitemap_robots import SitemapRobotsParser
        original_get_all = SitemapRobotsParser.get_all_urls

        def patched_get_all(self_inner):
            print(f"[crawler] Injecting {len(urls)} crawled URLs into scraper")
            return urls, None

        SitemapRobotsParser.get_all_urls = patched_get_all

        try:
            stats = scraper.scrape_website(max_pages=max_pages)
        finally:
            SitemapRobotsParser.get_all_urls = original_get_all
            db.print_stats()
            db.close()

        return stats

    # CLI: python crawler.py <domain> [max_pages] [max_depth]
    if len(sys.argv) < 2:
        print("Usage: python crawler.py <domain> [max_pages] [max_depth]")
        print("Example: python crawler.py crowdgen.com 200 3")
        sys.exit(1)

    domain_arg = sys.argv[1]
    max_pages_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    max_depth_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    crawl_and_scrape(domain_arg, max_pages_arg, max_depth_arg)
