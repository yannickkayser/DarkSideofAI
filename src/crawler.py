"""
crawler.py
==========
BFS (breadth-first search) link crawler for websites that do not publish
an XML sitemap.

Pipeline position:
  Stage 0 — URL Discovery (alternative to sitemap_robots.py)
  Runs BEFORE WebScraper when the --crawl flag is passed to main.py.
  Discovered URLs are injected into the scraper via monkey-patching
  (see main.py and the standalone entry point at the bottom of this file).

What this script does:
  Crawls a single website by following internal links starting from
  base_url.  Uses BFS so pages are discovered in order of link distance
  from the homepage.  Returns an ordered list of discovered URLs without
  scraping their content — content is scraped by scraper.py in a second
  pass.

Input:
  - base_url   : the website's homepage as configured in config/config.py
  - max_pages  : discovery stops once this many distinct URLs are found
  - max_depth  : pages reachable only through chains longer than this are
                 ignored (avoids deep pagination spirals)
  - rate_limit : seconds to wait between HTTP requests (politeness)

Output:
  - list of str  — normalised internal URLs discovered during crawl
  These are passed to WebScraper by monkey-patching
  SitemapRobotsParser.get_all_urls so the scraper uses them instead of
  querying the sitemap.  The list is not written to disk or the database
  directly; it is ephemeral.

When to use this vs sitemap mode:
  - Sitemap mode (default): site publishes /sitemap.xml or lists sitemaps
    in robots.txt.  Faster, more complete, less likely to miss pages.
  - Crawler mode (--crawl): no sitemap, or sitemap is incomplete.
    Slower but discovers pages that are only reachable via HTML links.
  Examples:
    Sitemap mode — appen.com, scale.com, labelbox.com
    Crawler mode — crowdgen.com, telusinternational.ai

Key design decisions:
  - BFS (not DFS): ensures shallow pages (high-traffic, most
    representative of the site's public-facing language) are collected
    first; useful when max_pages cuts off the crawl.
  - Fragment stripping & normalisation: removes #section anchors and
    trailing slashes so the same page is not visited twice under
    different URL spellings.
  - Extension exclusion: binary and media files (.pdf, .jpg, .css, .js,
    etc.) are filtered out — they contain no text for analysis.
  - robots.txt: not explicitly parsed here; rate_limit and max_depth are
    the primary politeness mechanisms.

Dependencies:
  - requests     : lightweight HTTP for link discovery
  - BeautifulSoup: HTML link extraction
  - collections.deque : BFS queue

Thesis note:
  The crawler mode and its BFS discovery strategy are documented in the
  Data Collection section of the methodology chapter as the fallback for
  sites without sitemaps.
"""

import time
from collections import deque
from urllib.parse import urlparse, urljoin, urldefrag
from typing import Set, List, Optional

import requests
from bs4 import BeautifulSoup


class LinkCrawler:
    """
    Crawls a website by following internal links in breadth-first order.

    Designed for sites without an XML sitemap (e.g. crowdgen.com,
    telusinternational.ai).  The discovered URL list is intended to be
    passed to WebScraper for the actual content extraction.

    Usage:
        crawler = LinkCrawler("https://crowdgen.com", max_pages=200, max_depth=3)
        urls = crawler.discover_urls()
        # Pass urls to WebScraper or use standalone

    Attributes:
        base_url  : starting URL, trailing slash stripped
        domain    : netloc of base_url (used for internal-link filtering)
        max_pages : upper bound on number of URLs in the returned list;
                    discovery stops once reached
        max_depth : maximum link-hop distance from base_url; pages only
                    reachable through deeper paths are skipped
        rate_limit: seconds to sleep between fetches (politeness)
        session   : requests.Session with User-Agent set
        exclude_extensions: file extensions that are excluded from crawling
                    (binary / media / script files with no text value)
    """

    def __init__(
        self,
        base_url: str,
        max_pages: int = 1000,
        max_depth: int = 8,
        rate_limit: float = 1.5,
        user_agent: str = "Mozilla/5.0 (compatible; ScrapeBot/1.0)",
        exclude_extensions: Optional[Set[str]] = None,
    ):
        """
        Initialise the crawler.

        Args:
            base_url  : Full URL of the site's homepage (e.g.
                        "https://crowdgen.com").  Trailing slashes are
                        stripped.
            max_pages : Hard cap on discovered URLs.  Crawl terminates
                        as soon as the visited set reaches this size.
                        Default 1000 is conservative; increase for large
                        sites.
            max_depth : Ignore pages reachable only through chains of
                        more than max_depth links.  Controls how deep
                        into the site structure the crawler descends.
                        Default 8 is permissive; set to 3-4 for focused
                        crawls of the top-level content.
            rate_limit: Seconds to sleep after each page fetch.  Lower
                        values are faster but more likely to trigger bot
                        detection or rate limits.  Use the value from the
                        site's config entry (config.get('rate_limit', 1.5)).
            user_agent: HTTP User-Agent string sent with each request.
                        The default declares a generic bot; some sites
                        require a browser-like string to respond.
            exclude_extensions: Set of lowercase file extensions to skip.
                        Defaults to a comprehensive list of binary and
                        non-text resources.  Override only to add extra
                        extensions, not to shrink the list.
        """
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
        """
        Return True if the URL belongs to the same domain as base_url.

        Only the netloc (host) component is compared, so subpaths and
        query strings are ignored.  This correctly excludes external
        links while including all paths under the crawled domain.
        """
        return urlparse(url).netloc == self.domain

    def _normalize(self, url: str) -> Optional[str]:
        """
        Normalise a URL for deduplication.

        Steps:
          1. Strip fragment (#section) — same page, different anchor
          2. Strip trailing slash — /about and /about/ are the same page
          3. Check extension — return None if file type is excluded

        Returns:
            Normalised URL string, or None if the URL should be skipped
            (excluded extension).
        """
        url, _ = urldefrag(url)
        url = url.rstrip("/") or "/"
        path = urlparse(url).path.lower()
        if any(path.endswith(ext) for ext in self.exclude_extensions):
            return None
        return url

    def _fetch_links(self, url: str) -> List[str]:
        """
        Fetch one page with requests and return all normalised internal hrefs.

        Uses requests (not Playwright) because link discovery only needs
        the raw HTML anchor tags — JavaScript rendering is not required
        for finding href attributes.  Playwright is reserved for the
        content-extraction pass in scraper.py.

        Args:
            url: Page URL to fetch.

        Returns:
            List of normalised internal URLs found on the page.
            Empty list on any error (connection, timeout, non-200 status).
        """
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

        Algorithm:
          - Initialise a deque with (base_url, depth=0)
          - Pop the front item; fetch its links; enqueue new internal
            links at depth+1 (if depth < max_depth and max_pages not
            reached)
          - Sleep rate_limit seconds between fetches
          - Stop when queue is empty or visited set hits max_pages

        The visited set is used both as a deduplication store and as the
        final result.  BFS ordering means pages closer to the homepage
        are collected first, which is preferable when max_pages truncates
        the crawl (shallow pages are more representative of the site's
        public language than deep pagination pages).

        Returns:
            Ordered list of all discovered internal URLs, including
            base_url.  Order reflects BFS discovery sequence — i.e.
            pages visited earlier are first.
            Returns an empty list only if the base URL itself is
            unreachable.
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
# Allows the crawler to be run directly:
#   python crawler.py crowdgen.com 200 3
#
# In the main pipeline, crawler.py is used indirectly via main.py with
# the --crawl flag, which calls scrape_website(domain, max_pages,
# use_crawler=True).  The standalone entry point is provided for quick
# manual testing and debugging of URL discovery without running the full
# scraper.
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
        """
        Standalone pipeline: discover URLs via BFS, then scrape content.

        This mirrors the logic in main.py's scrape_website(use_crawler=True)
        but runs as a self-contained script.

        Steps:
          1. Look up site config from config/config.py
          2. Run BFS LinkCrawler → list of URLs
          3. Open a Database connection (creates tables if needed)
          4. Monkey-patch SitemapRobotsParser.get_all_urls so WebScraper
             uses the crawled URLs instead of querying the sitemap
          5. Run WebScraper.scrape_website — persists pages/links to DB
          6. Restore the original method (finally block ensures this even
             on error)

        The monkey-patching pattern is used because WebScraper always
        calls SitemapRobotsParser to obtain its URL list.  Rather than
        adding a separate code path inside WebScraper, we temporarily
        replace the method so the existing scraping logic runs unchanged.

        Args:
            domain    : Domain as configured in config/config.py.
            max_pages : Maximum pages to scrape per run.
            max_depth : Maximum BFS depth for URL discovery.
        """
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
            # Always restore the original method — even if scraping crashes.
            # Leaving the monkey-patch active would break subsequent sitemap
            # mode calls in the same Python process.
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
