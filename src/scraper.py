"""
scraper.py
==========
Playwright-based content scraper for JavaScript-heavy websites.

Pipeline position:
  Stage 0 — Data Collection (runs alongside or after crawler.py for URL
  discovery; stores results that ALL downstream analysis scripts consume)
  Output: populates the `websites`, `pages`, and `links` tables in
          data/scraping.db.
  Also saves raw JSON backups to data/raw/<domain>/ for disaster recovery.

What this script does:
  Given a list of URLs (from sitemap_robots.py or monkey-patched in by
  crawler.py), visits each page in a Playwright Chromium browser, waits
  for JavaScript to finish rendering, and extracts:
    - Page title
    - Main text content (body/article/main element, scripts stripped)
    - Internal and external hyperlinks
    - CSS colour palette (background, text, link, button colours)

  Each page's text is stored in the `pages` table and each link in the
  `links` table.  A raw JSON backup is also written to disk.

Key design decisions:

  Playwright over requests/BeautifulSoup:
    All target sites use modern JavaScript frameworks (React, Framer,
    Next.js).  Their text content is injected by JS after page load, so
    plain HTTP GET responses are empty shells.  Playwright drives a real
    Chromium browser and waits for the DOM to hydrate.

  Stealth settings:
    navigator.webdriver is set to undefined, navigator.plugins is
    spoofed, and a real Mac/Chrome User-Agent is used.  Without these,
    some sites detect automation and serve empty or bot-detection pages.

  Fresh page per URL:
    A new browser page object is created for each URL rather than
    reusing tabs.  This prevents stale JS state from one page
    contaminating the next and reduces the chance of being blocked via
    session fingerprinting.

  Browser restart on crash:
    scale.com occasionally kills the Chromium process.  The scraper
    catches the resulting exception and relaunches the browser so the
    run continues without manual intervention.

  429 (rate-limit) handling:
    If the server returns HTTP 429 it reads the Retry-After header (or
    uses exponential backoff) before retrying.  This prevents the
    scraper from hammering a server that has signalled it is overloaded.

  CSS colour extraction:
    Collected as exploratory data for possible visual-register analysis.
    The colours are stored as JSON in pages.css_colors but are NOT used
    in the linguistic analysis pipeline (01_prepare.py through
    04_step2_export.py).

  networkidle + JS hydration wait:
    After navigation the scraper waits for (a) the network to go idle
    and (b) the body text to exceed 200 characters.  An extra 1.5 s
    sleep is added for Framer-based sites that inject content in a
    final micro-task after the network settles.

Dependencies:
  - playwright (sync API)
  - BeautifulSoup (HTML parsing after JS render)
  - config/config.py  (HEADLESS, PAGE_WAIT_TIMEOUT, etc.)
  - database.py       (Database class for DB writes)
  - sitemap_robots.py (SitemapRobotsParser — monkey-patched by crawler.py)
  - validator.py      (validate_page_data)
  - logger.py         (structured scrape logging)

Thesis note:
  The Playwright-based scraping methodology, stealth settings, rate
  limiting, and two-mode (sitemap / crawler) URL discovery are described
  in the Data Collection section of the methodology chapter.
"""

import time
import json
import re
import random
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
    """
    Scrapes JavaScript-heavy websites using a headless Playwright browser.

    Designed for sites in the DarkSideofAI corpus that render content via
    React / Framer / Next.js and therefore cannot be scraped with plain
    HTTP requests.

    Typical usage (via main.py):
        config = get_website_config("appen.com")
        db = Database(DB_PATH)
        scraper = WebScraper(config, db)
        stats = scraper.scrape_website(max_pages=200)
        db.close()

    The scraper reads its URL list from SitemapRobotsParser.get_all_urls
    at the start of scrape_website().  When crawler.py is used, that
    method is monkey-patched to return the BFS-discovered URL list
    instead of querying the actual sitemap.

    Attributes:
        config       : Site configuration dict from config/config.py.
                       Keys: base_url, name, type, audience, rate_limit,
                       max_depth.
        db           : Open Database instance (scraper does not open or
                       close the DB itself).
        base_url     : Site homepage URL.
        domain       : netloc of base_url.
        rate_limit   : Seconds between requests (from config or global
                       default).
        max_depth    : Maximum URL depth (from config or 3).
        logger       : Per-domain file logger.
        visited_urls : Set of URLs visited in the current run (in-memory
                       deduplication; db.page_exists provides persistence).
        stats        : Dict tracking pages_scraped, pages_failed,
                       total_bytes.
    """

    def __init__(self, website_config: dict, db: Database):
        """
        Args:
            website_config: Site configuration from config/config.py.
                            Required keys: base_url, name.
                            Optional: rate_limit (default RATE_LIMIT_DELAY),
                            max_depth (default 3).
            db:             Open Database connection.  The scraper uses
                            db.add_website, db.add_page, db.add_link,
                            db.page_exists, db.update_website_last_scraped.
        """
        self.config = website_config
        self.db = db
        self.base_url = website_config['base_url']
        self.domain = urlparse(self.base_url).netloc
        self.rate_limit = website_config.get('rate_limit', RATE_LIMIT_DELAY)
        self.max_depth = website_config.get('max_depth', 3)

        # Setup logger — one log file per domain so runs can be reviewed
        # independently without trawling a combined log
        self.logger = setup_logger(
            f"scraper.{self.domain}",
            f"logs/scraper_{self.domain}.log"
        )

        # In-memory visited set — prevents re-visiting within the same
        # Python process run; db.page_exists handles cross-run deduplication
        self.visited_urls = set()
        self.last_request_time = 0

        # Stats returned by scrape_website() for summary reporting
        self.stats = {
            'pages_scraped': 0,
            'pages_failed': 0,
            'total_bytes': 0
        }

    def _wait_for_rate_limit(self):
        """
        Enforce the per-site rate limit between requests.

        Calculates the time elapsed since the last request and sleeps
        for the remainder of the rate_limit window.  Uses wall-clock
        time rather than a fixed sleep so processing time counts toward
        the interval.
        """
        if self.last_request_time > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit:
                sleep_time = self.rate_limit - elapsed
                time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _extract_directory(self, url: str) -> str:
        """
        Extract the directory path component of a URL.

        Used to group pages by URL section in the database
        (pages.directory column).  For example:
          https://appen.com/solutions/data-annotation → /solutions

        Returns "/" for the homepage and for single-segment paths.
        """
        parsed = urlparse(url)
        path = parsed.path.rstrip('/')
        if not path or path == '/':
            return '/'
        return '/'.join(path.split('/')[:-1]) or '/'

    def _extract_css_colors(self, page: Page, soup) -> dict:
        """
        Extract CSS colour values from key element types on the page.

        Evaluates JavaScript in the live Playwright browser to read
        computed styles — something that cannot be done from the raw
        HTML alone because styles are often applied dynamically.

        Returns a dict with four lists:
          background_colors : from body, header, main, section, nav, footer
          text_colors       : from p, h1–h6, span
          link_colors       : from <a> elements
          button_colors     : from button, .btn, [role=button] — both
                             background and foreground colours

        Notes:
          - Transparent backgrounds (rgba(0,0,0,0)) are filtered out.
          - Each list is capped at 10–20 unique values for storage
            efficiency.
          - This data is stored in pages.css_colors as JSON but is NOT
            used in the downstream linguistic analysis pipeline (it was
            collected for exploratory visual-register analysis that was
            not pursued in the final thesis).
          - On any JS evaluation error the method returns four empty
            lists and logs a warning.

        Args:
            page : Live Playwright Page object (browser tab).
            soup : BeautifulSoup of the page HTML (not currently used;
                   kept for potential future fallback parsing).

        Returns:
            Dict with keys background_colors, text_colors, link_colors,
            button_colors.
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
            # Return empty lists on error — missing colour data does not
            # affect the text analysis pipeline
            colors = {
                'background_colors': [],
                'text_colors': [],
                'link_colors': [],
                'button_colors': []
            }

        return colors

    def _extract_text_from_page(self, page: Page) -> Dict[str, any]:
        """
        Extract text content, metadata, and links from a fully-rendered page.

        Execution sequence:
          1. Wait for network idle (JS assets loaded)
          2. Wait for body text to exceed 200 characters (DOM hydrated)
          3. Sleep 1.5 s for late-rendering Framer sites
          4. Get full HTML from Playwright
          5. Parse with BeautifulSoup
          6. Extract CSS colours via JS evaluation
          7. Extract title (<title> tag, fallback to <h1>)
          8. Find the main content element (main > article > [role=main]
             > .content > #content > body — first match wins)
          9. Strip <script>, <style>, <noscript> from content element
         10. Extract all <a href> links, classify as internal/external

        The content element selection tries to isolate the page's main
        body text from navigation, header, and footer boilerplate.  This
        reduces noise in the corpus but is imperfect; residual boilerplate
        is handled later by the excluded_terms pipeline in
        01_prepare_additions.py.

        Args:
            page : Live Playwright Page object with a fully-loaded URL.

        Returns:
            Dict with keys:
              url            : Final URL after any redirects
              title          : Page title string
              text_content   : Plain text extracted from main content area
              html_element   : Tag name of the content container used
              directory      : URL directory path (from _extract_directory)
              content_length : Byte length of the raw HTML response
              css_colors     : Dict from _extract_css_colors
              links          : List of {url, anchor_text, type} dicts
        """
        # Wait for page to be fully loaded
        # FIX - wait for network idle, then wait for actual content to appear
        try:
            page.wait_for_load_state('networkidle', timeout=NETWORK_IDLE_TIMEOUT)
        except:
            pass  # Continue even if timeout


        # Wait for Framer/JS frameworks to finish hydrating the DOM
        try:
            page.wait_for_function(
                "() => document.body && document.body.innerText.trim().length > 200",
                timeout=15000
            )
        except:
            pass  # Use whatever is rendered so far

        # Extra pause for late-rendering elements (Framer injects after JS resolves)
        time.sleep(1.5)

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

        # Extract main content areas — try semantic selectors first,
        # fall back to body.  This prioritises article text over nav/
        # footer boilerplate.
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
            # Remove script and style elements — these add JavaScript
            # code and CSS rules to the text, not readable prose
            for script in main_content(['script', 'style', 'noscript']):
                script.decompose()

            text_content = main_content.get_text(separator=' ', strip=True)
            html_element = main_content.name
        else:
            text_content = ''
            html_element = 'body'

        # Extract all links for the links table
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
        """
        Write raw page data as a JSON file to data/raw/<domain>/.

        Provides a disaster-recovery backup separate from the SQLite
        database.  File names are <sha256_prefix>_<timestamp>.json to
        guarantee uniqueness without collisions.

        The raw data includes the full text_content, css_colors, and
        links, which allows re-importing without re-scraping if the
        database is lost or corrupted.

        Args:
            page_data: Dict returned by _extract_text_from_page.
        """
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
        """
        Scrape a single page: navigate, render, extract, validate.

        Retry logic:
          - Retries up to RETRY_ATTEMPTS times on recoverable errors
          - Uses exponential backoff (RETRY_DELAY * attempt) between
            retries
          - 429 responses trigger a Retry-After-aware wait before the
            next attempt
          - "closed" / "target" errors (browser crash) are non-retryable
            and return None immediately

        Validation:
          validate_page_data is called on the extracted data.  If
          validation fails the page_data is still returned (with a
          placeholder content string) so the URL is persisted to the DB
          and is not retried on future runs.  This prevents infinite
          re-scraping of pages that are legitimately empty.

        Args:
            page  : Open Playwright Page object (browser tab).
            url   : URL to navigate to.
            depth : Link depth from the homepage (0 for sitemap URLs).

        Returns:
            Dict of page data (keys as per _extract_text_from_page), or
            None on non-recoverable failure.
        """
        log_scrape_start(self.logger, url)

        for attempt in range(RETRY_ATTEMPTS):
            try:
                self._wait_for_rate_limit()

                # Check if page is valid
                if page.is_closed():
                    self.logger.error("Page is closed, cannot scrape")
                    return None

                # Navigate to page
                response = page.goto(url, wait_until='load', timeout=PAGE_WAIT_TIMEOUT)

                if not response:
                    raise Exception("No response from page")

                status_code = response.status

                if status_code == 429:
                    retry_after = response.headers.get('Retry-After')
                    wait = int(retry_after) if retry_after and retry_after.isdigit() else 30 * (2 ** attempt)
                    self.logger.warning(f"429 on {url} — waiting {wait}s before retry")
                    time.sleep(wait)
                    continue

                # Random human-like pause to avoid bot detection timing patterns
                time.sleep(random.uniform(1.5, 3.5))

                # Extract content
                page_data = self._extract_text_from_page(page)
                page_data['status_code'] = status_code
                page_data['depth'] = depth

                # Validate
                is_valid, message = validate_page_data(page_data)
                log_validation_result(self.logger, is_valid, f"{url}: {message}")

                if not is_valid:
                    self.logger.warning(f"Validation failed for {url}: {message}")
                    # Still return data so the URL gets saved to DB and isn't retried forever
                    page_data['text_content'] = page_data.get('text_content') or '[content unavailable]'
                    return page_data

                log_scrape_success(self.logger, url, status_code, page_data['content_length'])

                return page_data

            except KeyboardInterrupt:
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
        Scrape an entire website using its URL list.

        URL source: calls SitemapRobotsParser.get_all_urls.  In sitemap
        mode this queries the actual XML sitemap.  In crawler mode this
        method is monkey-patched by crawler.py to return the BFS URL
        list instead.  The scraping logic is identical in both cases.

        Browser lifecycle:
          - Launches Chromium with stealth args
          - Creates a context with Mac/Chrome User-Agent, viewport,
            locale, and timezone to mimic a real user
          - Adds an init script that removes navigator.webdriver and
            other automation indicators
          - Opens a new tab per URL (reduces state leakage between pages)
          - Restarts the entire browser if it crashes (scale.com issue)
          - Closes context and browser in a finally block

        DB writes per page (if successful):
          - db.add_page     : inserts into pages table (or upserts on
                              url_hash collision for idempotency)
          - db.add_link     : inserts each link found on the page

        Post-run:
          - Calls db.update_website_last_scraped to timestamp the run
          - Prints a summary of pages_scraped, pages_failed, total_bytes

        Args:
            max_pages: If set, stop after this many pages have been
                       successfully scraped.  None means scrape all
                       discovered URLs.

        Returns:
            stats dict with keys: pages_scraped, pages_failed,
            total_bytes.
        """
        self.logger.info(f"Starting scrape of {self.config['name']}")

        # Get sitemap URLs (or monkey-patched crawled URLs)
        parser = SitemapRobotsParser(self.base_url)
        sitemap_urls, crawl_delay = parser.get_all_urls()

        if crawl_delay and crawl_delay > self.rate_limit:
            self.logger.info(f"Using crawl delay from robots.txt: {crawl_delay}s")
            self.rate_limit = crawl_delay

        if not sitemap_urls:
            self.logger.warning("No sitemap found, scraping only base URL")
            sitemap_urls = [self.base_url]

        # Add website to database (upsert — safe to call on re-runs)
        website_id = self.db.add_website(
            domain=self.domain,
            name=self.config['name'],
            base_url=self.base_url,
            website_type=self.config.get('type', 'Unknown')
        )

        ## Start Playwright
        with sync_playwright() as p:
            # Browser args disable automation-detection features and
            # set a realistic window size
            browser = p.chromium.launch(
                headless=HEADLESS,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--window-size=1920,1080'
                ]
            )

            # Browser context with a realistic Mac/Chrome fingerprint.
            # locale and timezone_id are set so sites that serve different
            # content to non-US visitors see a US English user.
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

            # Stealth init script: overwrite browser properties that
            # standard automation detection checks.  Must be added to
            # context (not page) so it runs on every new page.
            context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                window.chrome = { runtime: {} };
                Object.defineProperty(navigator, 'permissions', {
                    get: () => ({ query: () => Promise.resolve({ state: 'granted' }) })
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

                    # Skip if already visited (in-memory or in DB from a
                    # prior run — provides idempotency across restarts)
                    if url in self.visited_urls or self.db.page_exists(url):
                        self.logger.debug(f"Skipping already scraped: {url}")
                        continue

                    self.visited_urls.add(url)

                    # Use a fresh page per URL to avoid stale state triggering bot detection
                    # Relaunch browser if it crashed (scale.com can kill the whole browser)
                    try:
                        page = context.new_page()
                    except Exception:
                        self.logger.warning("Browser died, relaunching...")
                        try:
                            browser.close()
                        except Exception:
                            pass
                        browser = p.chromium.launch(
                            headless=HEADLESS,
                            args=['--disable-blink-features=AutomationControlled',
                                '--disable-dev-shm-usage', '--no-sandbox',
                                '--disable-web-security',
                                '--window-size=1920,1080']
                        )
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
                        context.add_init_script("""
                            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
                            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                            window.chrome = { runtime: {} };
                            Object.defineProperty(navigator, 'permissions', {
                                get: () => ({ query: () => Promise.resolve({ state: 'granted' }) })
                            });
                        """)
                        context.set_default_timeout(PAGE_WAIT_TIMEOUT)
                        page = context.new_page()

                    # Scrape page
                    try:
                        page_data = self.scrape_page(page, url, depth=0)
                    except (RuntimeError, Exception) as e:
                        self.logger.warning(f"Failed {url}: {e}")
                        self.stats['pages_failed'] += 1
                        page_data = None
                    finally:
                        try:
                            page.close()
                        except Exception:
                            pass
                        except KeyboardInterrupt:
                            self.logger.warning("Scraping interrupted by user")

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

            except KeyboardInterrupt:
                self.logger.warning("Scraping interrupted by user")
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
            finally:
                try:
                    context.close()
                    browser.close()
                except:
                    pass  #

        # Update website
        self.db.update_website_last_scraped(website_id)

        # Print summary
        self._print_summary()

        return self.stats

    def _print_summary(self):
        """Print scraping summary to stdout at end of a run."""
        print("\n\n" + "="*60)
        print(f"SCRAPING SUMMARY - {self.config['name']}")
        print("="*60)
        print(f"Pages Scraped: {self.stats['pages_scraped']}")
        print(f"Pages Failed: {self.stats['pages_failed']}")
        print(f"Total Data: {self.stats['total_bytes'] / (1024*1024):.2f} MB")
        print("="*60 + "\n")
