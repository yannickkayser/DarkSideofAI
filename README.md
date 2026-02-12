# Simple Web Scraper

A clean, efficient web scraper for JavaScript-heavy websites with comprehensive logging and validation.

## Features

✅ **Playwright-based** - Handles JavaScript-heavy sites  
✅ **Sitemap parsing** - Automatically finds and uses sitemap.xml  
✅ **Robots.txt compliance** - Respects crawl delays  
✅ **Rich metadata** - Stores URL, title, text, directory, HTML element, links  
✅ **SQLite database** - Simple, portable storage  
✅ **Comprehensive logging** - Track all operations  
✅ **Built-in validation** - Ensures data quality  

## Project Structure

```
scraper_simple/
├── config.py              # All configuration in one place
├── database.py            # SQLite database operations
├── scraper.py             # Main scraping engine (Playwright)
├── sitemap_robots.py      # Sitemap and robots.txt parsing
├── validator.py           # Validation functions
├── logger.py              # Logging setup
├── main.py                # Entry point
├── requirements.txt       # Dependencies
├── data/                  # Database storage (auto-created)
└── logs/                  # Log files (auto-created)
```

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Playwright browsers**:
   ```bash
   playwright install chromium
   ```

## Quick Start

### Interactive Mode
```bash
python main.py
```

### Command Line

**Scrape single website**:
```bash
python main.py mindrift.ai
```

**Scrape with page limit**:
```bash
python main.py mindrift.ai 20
```

**Scrape all websites**:
```bash
python main.py --all
```

**Show statistics**:
```bash
python main.py --stats
```

## Configured Websites

Currently configured websites (edit `config.py` to add more):
- mindrift.ai
- crowdgen.com
- appen.com

## Database Schema

The SQLite database stores:

### websites
- domain, name, base_url, website_type
- created_at, last_scraped

### pages
- url, title, text_content
- directory, html_element, page_depth
- status_code, content_length
- website_id (foreign key)

### links
- source_page_id, target_url
- anchor_text, link_type (internal/external)

## How It Works

1. **Parse robots.txt** - Get crawl delay and find sitemap
2. **Parse sitemap.xml** - Extract all URLs
3. **Scrape pages** - Use Playwright to render JavaScript
4. **Extract content** - Get text from main content areas
5. **Store data** - Save to SQLite with metadata
6. **Log & validate** - Track operations and validate data

## Configuration

Edit `config.py` to:
- Add new websites
- Adjust rate limits
- Change timeout settings
- Modify logging level

Example:
```python
WEBSITES = {
    "example.com": {
        "name": "Example Site",
        "base_url": "https://example.com/",
        "type": "Example Type",
        "rate_limit": 2.0,
        "max_depth": 3,
    },
}
```

## Logging

Logs are saved to `logs/` directory:
- `scraper_{domain}.log` - Per-website logs
- All operations are logged with timestamps

## Validation

Built-in validators check:
- URL format
- Content length (minimum 50 chars)
- Required fields (url, title, text_content)
- Sitemap and robots.txt format

## Accessing Data

```python
from database import Database
from config import DB_PATH

db = Database(str(DB_PATH))

# Get statistics
stats = db.get_stats()
print(f"Total pages: {stats['total_pages']}")

# Check if URL exists
exists = db.page_exists("https://example.com/page")

db.close()
```

## Customization

**Change rate limit**:
Edit `config.py` → `RATE_LIMIT_DELAY`

**Change logging level**:
Edit `config.py` → `LOG_LEVEL` (DEBUG, INFO, WARNING, ERROR)

**Add new website**:
Edit `config.py` → Add to `WEBSITES` dictionary

**Adjust timeouts**:
Edit `config.py` → `PAGE_WAIT_TIMEOUT`, `NETWORK_IDLE_TIMEOUT`

## Tips

- Start with small page limits (`python main.py mindrift.ai 5`)
- Check logs if scraping fails
- Use `--stats` to monitor progress
- Database is in `data/scraping.db`

## Troubleshooting

**No pages scraped**:
- Check if sitemap exists for the website
- Look at logs for errors
- Try increasing timeouts

**Validation failures**:
- Pages may have little content
- Check minimum content length in validator.py

**Playwright errors**:
- Run `playwright install` again
- Check if browsers are installed

## Next Steps

To extend this scraper:
1. Add custom content extractors
2. Implement more sophisticated link following
3. Add data export (CSV, JSON)
4. Create analysis scripts
5. Add scheduling/cron support
