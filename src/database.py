"""
Simple SQLite database for storing scraped content with metadata
"""
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


class Database:
    """Simple SQLite database manager"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.init_database()
    
    def connect(self):
        """Connect to database"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def init_database(self):
        """Create tables if they don't exist"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Websites table
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
        
        # Pages table with rich metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                website_id INTEGER,
                url TEXT UNIQUE NOT NULL,
                url_hash TEXT UNIQUE NOT NULL,
                title TEXT,
                text_content TEXT,
                directory TEXT,
                html_element TEXT,
                page_depth INTEGER,
                status_code INTEGER,
                content_length INTEGER,
                css_colors TEXT,  
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (website_id) REFERENCES websites(id)
            )
        """)
        
        # Links table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_page_id INTEGER,
                target_url TEXT,
                anchor_text TEXT,
                link_type TEXT,
                FOREIGN KEY (source_page_id) REFERENCES pages(id)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pages_website ON pages(website_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pages_url_hash ON pages(url_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_page_id)")
        
        conn.commit()
    
    def add_website(self, domain: str, name: str, base_url: str, 
                    website_type: str) -> int:
        """Add or get website"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Try to get existing
        cursor.execute("SELECT id FROM websites WHERE domain = ?", (domain,))
        row = cursor.fetchone()
        
        if row:
            return row[0]
        
        # Insert new
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
        """Add page with metadata"""
        conn = self.connect()
        cursor = conn.cursor()
        
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        
        # Check if exists
        cursor.execute("SELECT id FROM pages WHERE url_hash = ?", (url_hash,))
        if cursor.fetchone():
            return None  # Already exists
        
        cursor.execute("""
            INSERT INTO pages (
                website_id, url, url_hash, title, text_content,
                directory, html_element, page_depth, status_code, content_length, css_colors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (website_id, url, url_hash, title, text_content, 
              directory, html_element, depth, status_code, content_length, css_colors))
        
        conn.commit()
        return cursor.lastrowid
    
    def add_link(self, source_page_id: int, target_url: str, 
                 anchor_text: str, link_type: str):
        """Add link between pages"""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO links (source_page_id, target_url, anchor_text, link_type)
            VALUES (?, ?, ?, ?)
        """, (source_page_id, target_url, anchor_text, link_type))
        
        conn.commit()
    
    def page_exists(self, url: str) -> bool:
        """Check if URL already scraped"""
        conn = self.connect()
        cursor = conn.cursor()
        
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        cursor.execute("SELECT 1 FROM pages WHERE url_hash = ?", (url_hash,))
        
        return cursor.fetchone() is not None
    
    def update_website_last_scraped(self, website_id: int):
        """Update last scraped timestamp"""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE websites 
            SET last_scraped = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (website_id,))
        
        conn.commit()
    
    def get_stats(self, website_id: Optional[int] = None) -> Dict[str, Any]:
        """Get scraping statistics"""
        conn = self.connect()
        cursor = conn.cursor()
        
        if website_id:
            cursor.execute("""
                SELECT COUNT(*) as total_pages,
                       SUM(content_length) as total_bytes
                FROM pages WHERE website_id = ?
            """, (website_id,))
        else:
            cursor.execute("""
                SELECT COUNT(*) as total_pages,
                       SUM(content_length) as total_bytes
                FROM pages
            """)
        
        row = cursor.fetchone()
        
        return {
            'total_pages': row['total_pages'] or 0,
            'total_bytes': row['total_bytes'] or 0,
            'total_mb': (row['total_bytes'] or 0) / (1024 * 1024)
        }
    
    def print_stats(self):
        """Print database statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("DATABASE STATISTICS")
        print("="*60)
        print(f"Total Pages: {stats['total_pages']}")
        print(f"Total Data: {stats['total_mb']:.2f} MB")
        print("="*60 + "\n")
