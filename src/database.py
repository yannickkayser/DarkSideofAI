"""
database.py
===========
SQLite database manager for the web scraping layer.

Pipeline position:
  Stage 0 — Infrastructure
  Used exclusively by: main.py, scraper.py, crawler.py
  Does NOT interact with the analysis pipeline (01_prepare onwards).

What this module does:
  Wraps a single SQLite connection and provides the minimal CRUD operations
  the scraper needs to persist raw page content.  Three tables are created
  on first use:

    websites  — one row per scraped domain (name, base_url, type)
    pages     — one row per scraped URL (raw text_content, CSS colors, links)
    links     — outgoing hrefs from each page (used for deduplication)

  All analysis scripts downstream read from pages via the corpus_view
  created by 01_prepare.py; they do NOT use this Database class directly.

Key design decisions:
  - url_hash (SHA-256) is the deduplication key on pages, not the URL
    string itself.  This prevents near-duplicate URLs from being scraped
    twice even if query parameters differ slightly.
  - ON CONFLICT … DO UPDATE on pages means re-scraping a URL refreshes
    its content rather than inserting a duplicate row.
  - The Database class holds a single persistent connection (self.conn).
    This is intentional — the scraper is single-threaded, and persistent
    connections avoid per-request reconnect overhead.

Thesis section: not directly cited — this is data-collection infrastructure.
  The table schemas appear in the methodological appendix describing how the
  raw corpus was assembled before preprocessing.
"""

import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


class Database:
    """
    Lightweight SQLite manager for the scraping pipeline.

    Usage:
        db = Database("data/scraping.db")
        website_id = db.add_website(domain, name, base_url, website_type)
        page_id = db.add_page(website_id, url, title, text_content, ...)
        db.close()
    """

    def __init__(self, db_path: str):
        """
        Initialise the database, creating the file and tables if they don't exist.

        Args:
            db_path: Absolute or relative path to the SQLite file.
                     Parent directories are created automatically.
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.init_database()

    def connect(self):
        """Open the connection lazily; return existing connection if already open."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        """Close the connection and release the file lock."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def init_database(self):
        """
        Create tables and indexes if they do not already exist.

        Schema notes:
          - websites.domain is UNIQUE — re-scraping the same domain updates
            last_scraped rather than creating duplicate rows.
          - pages.url_hash (SHA-256) is the deduplication key; the full URL
            string is also stored for diagnostics and corpus_view joins.
          - links stores the outgoing hyperlinks found on each page.
            These are used by find_duplicates.py and for coverage diagnostics,
            not by the linguistic analysis pipeline.
          - css_colors is stored as a JSON string (background_colors,
            text_colors, link_colors, button_colors); not currently used in
            the linguistic analysis but retained for potential future study
            of visual register differences.
        """
        conn = self.connect()
        cursor = conn.cursor()

        # Websites table: one row per scraped domain
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS websites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT UNIQUE NOT NULL,
                name TEXT,
                base_url TEXT,
                website_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_scraped TIMESTAMP
            )
        """)

        # Pages table: one row per scraped URL — this is the primary corpus store
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                website_id INTEGER,
                url TEXT UNIQUE NOT NULL,
                url_hash TEXT UNIQUE NOT NULL,   -- SHA-256 of URL, deduplication key
                title TEXT,
                text_content TEXT,               -- main body text after JS render
                directory TEXT,                  -- URL path prefix, used for section analysis
                html_element TEXT,               -- tag that contained main content (main/article/body)
                page_depth INTEGER,              -- crawl depth (0 = base URL)
                status_code INTEGER,
                content_length INTEGER,          -- raw HTML byte length
                css_colors TEXT,                 -- JSON: {background_colors, text_colors, ...}
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (website_id) REFERENCES websites(id)
            )
        """)

        # Links table: outgoing hrefs from each page
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_page_id INTEGER,
                target_url TEXT,
                anchor_text TEXT,
                link_type TEXT,   -- 'internal' or 'external'
                FOREIGN KEY (source_page_id) REFERENCES pages(id)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pages_website ON pages(website_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pages_url_hash ON pages(url_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_page_id)")

        conn.commit()

    def add_website(self, domain: str, name: str, base_url: str,
                    website_type: str) -> int:
        """
        Upsert a website record and return its id.

        If the domain already exists (from a previous scrape run), returns
        the existing id without modifying the record.  This means the
        last_scraped timestamp is only updated via update_website_last_scraped().

        Args:
            domain:       Bare domain string, e.g. 'appen.com'
            name:         Human-readable name from config
            base_url:     Full base URL including scheme
            website_type: Free-text type from config (e.g. 'Algorithmic Crowd Market')

        Returns:
            Integer primary key of the website row.
        """
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM websites WHERE domain = ?", (domain,))
        row = cursor.fetchone()
        if row:
            return row[0]

        cursor.execute("""
            INSERT INTO websites (domain, name, base_url, website_type)
            VALUES (?, ?, ?, ?)
        """, (domain, name, base_url, website_type))
        conn.commit()
        return cursor.lastrowid

    def add_page(self, website_id: int, url: str, title: str,
                 text_content: str, directory: str, html_element: str,
                 depth: int, status_code: int, content_length: int,
                 css_colors: str = None) -> int:
        """
        Upsert a page record and return its id.

        Uses INSERT OR REPLACE with ON CONFLICT to update existing records
        when a URL is re-scraped (e.g. after site updates).

        Args:
            website_id:     FK to websites.id
            url:            Full URL of the page
            title:          Page <title> or <h1> text
            text_content:   Extracted body text (after JS render, script/style removed)
            directory:      URL directory path prefix (e.g. '/solutions')
            html_element:   HTML tag that contained main content ('main', 'article', 'body')
            depth:          BFS crawl depth (0 = base URL from sitemap)
            status_code:    HTTP response code
            content_length: Raw HTML byte length
            css_colors:     JSON string of extracted CSS color values (optional)

        Returns:
            Integer primary key of the page row.
        """
        conn = self.connect()
        cursor = conn.cursor()

        url_hash = hashlib.sha256(url.encode()).hexdigest()

        cursor.execute("""
            INSERT INTO pages (
                website_id, url, url_hash, title, text_content,
                directory, html_element, page_depth, status_code, content_length, css_colors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url_hash) DO UPDATE SET
                title = excluded.title,
                text_content = excluded.text_content,
                scraped_at = CURRENT_TIMESTAMP,
                status_code = excluded.status_code,
                content_length = excluded.content_length,
                css_colors = excluded.css_colors
        """, (website_id, url, url_hash, title, text_content,
              directory, html_element, depth, status_code, content_length, css_colors))

        conn.commit()
        cursor.execute("SELECT id FROM pages WHERE url_hash = ?", (url_hash,))
        return cursor.fetchone()[0]

    def add_link(self, source_page_id: int, target_url: str,
                 anchor_text: str, link_type: str):
        """
        Persist a link found on source_page_id.

        Deletes and re-inserts links for the source page on each call —
        this means re-scraping a page also refreshes its outgoing links.

        Args:
            source_page_id: FK to pages.id
            target_url:     Full target URL
            anchor_text:    Visible link text
            link_type:      'internal' (same domain) or 'external'
        """
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM links WHERE source_page_id = ?", (source_page_id,))
        cursor.execute("""
            INSERT INTO links (source_page_id, target_url, anchor_text, link_type)
            VALUES (?, ?, ?, ?)
        """, (source_page_id, target_url, anchor_text, link_type))

        conn.commit()

    def page_exists(self, url: str) -> bool:
        """
        Check whether a URL has already been scraped.
        Uses the SHA-256 hash for O(1) lookup rather than a string scan.

        Args:
            url: Full URL to check.

        Returns:
            True if the URL exists in the pages table.
        """
        conn = self.connect()
        cursor = conn.cursor()

        url_hash = hashlib.sha256(url.encode()).hexdigest()
        cursor.execute("SELECT 1 FROM pages WHERE url_hash = ?", (url_hash,))
        return cursor.fetchone() is not None

    def update_website_last_scraped(self, website_id: int):
        """
        Update the last_scraped timestamp for a website.
        Called by WebScraper.scrape_website() after completing a scrape run.
        """
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE websites
            SET last_scraped = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (website_id,))
        conn.commit()

    def get_stats(self, website_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Return total page count and total scraped bytes.

        Args:
            website_id: If provided, scope stats to a single website.
                        If None, return corpus-wide totals.

        Returns:
            Dict with keys: total_pages (int), total_bytes (int), total_mb (float).
        """
        conn = self.connect()
        cursor = conn.cursor()

        if website_id:
            cursor.execute("""
                SELECT COUNT(*) as total_pages, SUM(content_length) as total_bytes
                FROM pages WHERE website_id = ?
            """, (website_id,))
        else:
            cursor.execute("""
                SELECT COUNT(*) as total_pages, SUM(content_length) as total_bytes
                FROM pages
            """)

        row = cursor.fetchone()
        return {
            'total_pages': row['total_pages'] or 0,
            'total_bytes': row['total_bytes'] or 0,
            'total_mb': (row['total_bytes'] or 0) / (1024 * 1024)
        }

    def print_stats(self):
        """Print corpus-wide scraping statistics to stdout."""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("DATABASE STATISTICS")
        print("="*60)
        print(f"Total Pages: {stats['total_pages']}")
        print(f"Total Data: {stats['total_mb']:.2f} MB")
        print("="*60 + "\n")
