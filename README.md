# Web Scraping Project

A robust, well-organized web scraping framework for extracting and storing content from multiple websites.

## Features

- ✅ **Modular Architecture**: Separate scrapers for static and dynamic websites
- ✅ **SQLite Database**: Store metadata and relationships
- ✅ **JSON Storage**: Raw content saved as structured JSON files
- ✅ **Comprehensive Metadata**: Track URLs, timestamps, content types, and more
- ✅ **Built-in Validation**: Automatic content and URL validation
- ✅ **Rate Limiting**: Respectful scraping with configurable delays
- ✅ **Error Handling**: Retry logic and comprehensive error logging
- ✅ **Progress Tracking**: Real-time feedback and statistics
- ✅ **Flexible Configuration**: Per-website configurations with selectors and rules

## Project Structure

```
scraping_project/
├── config/                    # Configuration files
│   ├── settings.py           # Global settings
│   ├── scraper_configs.py    # Website-specific configs
│   └── database_schema.py    # Database models
├── scrapers/                  # Scraper implementations
│   ├── base_scraper.py       # Base class with common functionality
│   ├── static_scraper.py     # For HTML-only sites
│   └── dynamic_scraper.py    # For JavaScript-heavy sites (to be implemented)
├── utils/                     # Utility functions
│   ├── logger.py             # Logging setup
│   ├── validation.py         # URL and content validation
│   ├── database.py           # Database operations
│   └── file_handler.py       # File I/O operations
├── data/                      # Data storage (auto-created)
│   ├── raw/                  # JSON files organized by domain
│   ├── media/                # Downloaded images, videos, PDFs
│   └── scraping.db           # SQLite database
├── logs/                      # Log files (auto-created)
├── tests/                     # Unit tests (to be implemented)
├── main.py                    # Entry point
└── requirements.txt           # Python dependencies
```

## Installation

1. **Clone or download the project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the project**:
   ```bash
   python main.py --init
   ```

## Quick Start

### Interactive Mode

Run the interactive menu:
```bash
python main.py --interactive
```

### Command Line Mode

**Scrape a website**:
```bash
python main.py --scrape mindrift.ai
```

**Scrape with page limit**:
```bash
python main.py --scrape mindrift.ai --max-pages 10
```

**Check project status**:
```bash
python main.py --status
```

**List configured websites**:
```bash
python main.py --list
```

## Configuration

### Adding a New Website

Edit `config/scraper_configs.py` and add a new configuration:

```python
"example.com": {
    "base_url": "https://example.com/",
    "name": "Example Site",
    "scraper_type": "static",  # or "dynamic"
    "allowed_domains": ["example.com"],
    "max_depth": 3,
    "rate_limit": 1.5,
    
    # CSS selectors for content extraction
    "selectors": {
        "main_content": "main, article",
        "title": "h1",
        "description": "meta[name='description']",
        "links": "a[href]",
        "images": "img[src]",
    },
    
    # URL patterns
    "include_patterns": [r"^https://example\.com/.*"],
    "exclude_patterns": [r".*\.(pdf|zip)$"],
    
    # Page classification
    "page_types": {
        "product": [r".*/products/.*"],
        "blog": [r".*/blog/.*"],
    },
}
```

### Adjusting Global Settings

Edit `config/settings.py`:

```python
# Rate limiting
RATE_LIMIT_DELAY = 1.5  # seconds between requests

# Retry behavior
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2

# Content limits
MAX_PAGE_SIZE = 10 * 1024 * 1024  # 10MB

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
```

## Database Schema

The project uses SQLAlchemy ORM with the following main tables:

- **websites**: Stores website metadata
- **scrape_sessions**: Tracks individual scraping sessions
- **pages**: Individual scraped pages with content references
- **links**: Extracted links and their relationships
- **media**: Images, videos, and other media files

## Usage Examples

### Basic Scraping

```python
from config import get_scraper_config, DB_PATH
from scrapers import StaticScraper

# Get configuration
config = get_scraper_config("mindrift.ai")

# Create scraper
scraper = StaticScraper(config, str(DB_PATH))

# Scrape website
result = scraper.scrape_website(max_pages=50)
```

### Accessing Data

```python
from utils import DatabaseManager
from config import DB_PATH

# Query database
with DatabaseManager(str(DB_PATH)) as db:
    # Get all pages from a website
    website = db.get_website("mindrift.ai")
    pages = db.session.query(Page).filter_by(website_id=website.id).all()
    
    for page in pages:
        print(f"{page.title}: {page.url}")
```

### Loading Scraped Content

```python
from utils import FileHandler

file_handler = FileHandler()

# Load page content from JSON
content = file_handler.load_page_content("mindrift.ai/abc123_20240101.json")
print(content['title'])
print(content['text_content'])
```

## Validation & Feedback

The project includes built-in validation and feedback functions:

```python
from utils import check_database_status, check_file_structure

# Check database status
check_database_status(str(DB_PATH))

# Check file structure
check_file_structure()

# Validation example
from utils import validate_url

result = validate_url("https://example.com")
if result['is_valid']:
    print("Valid URL!")
else:
    print(f"Errors: {result['errors']}")
```

## Metadata Captured

For each scraped page, the following metadata is captured:

**URL Information**:
- URL, URL hash
- Parent URL, depth in site hierarchy
- Domain

**Content Metadata**:
- Title, description
- Page type classification
- Language
- Full text content
- Structured content (headings, paragraphs, tables, code blocks)

**Technical Metadata**:
- HTTP status code
- Content type and length
- Response time
- Last modified header

**Timestamps**:
- Scraped timestamp
- Last modified
- Content updated timestamp

**Links & Media**:
- Internal and external links with anchor text
- Images with alt text and dimensions
- Videos and embedded media

## Logging

Logs are automatically created in the `logs/` directory:

- `scraping.log`: Main project log
- Individual module logs for detailed debugging

Configure logging level in `config/settings.py`:
```python
LOG_LEVEL = "INFO"  # DEBUG for verbose output
```

## Best Practices

1. **Always check robots.txt** before scraping a website
2. **Respect rate limits** - adjust `RATE_LIMIT_DELAY` appropriately
3. **Test with small samples** first (`--max-pages 5`)
4. **Monitor logs** for errors and warnings
5. **Back up data regularly** from the `data/` directory
6. **Validate content** - check `check_database_status()` after scraping

## Troubleshooting

**Issue**: Pages not being scraped
- Check URL patterns in `include_patterns` and `exclude_patterns`
- Verify selectors match the actual HTML structure
- Check logs for validation errors

**Issue**: No content extracted
- Verify CSS selectors in the config
- Check if the site requires JavaScript (use dynamic scraper)
- Look for validation errors in logs

**Issue**: Database errors
- Ensure database is initialized (`python main.py --init`)
- Check write permissions in `data/` directory

## Next Steps

To extend this project:

1. **Implement Dynamic Scraper**: Add `scrapers/dynamic_scraper.py` using Selenium/Playwright
2. **Add Tests**: Create unit tests in `tests/` directory
3. **Data Export**: Add functions to export data to CSV, Excel, etc.
4. **Analysis Tools**: Build analysis scripts using the structured data
5. **Scheduling**: Add cron jobs or scheduling for regular scraping
6. **API**: Create a REST API to query scraped data

## Example: Scraping mindrift.ai

```bash
# Initialize project
python main.py --init

# Start scraping
python main.py --scrape mindrift.ai --max-pages 20

# Check results
python main.py --status
```

The scraped data will be in:
- `data/raw/mindrift.ai/*.json` - Raw content files
- `data/scraping.db` - Metadata and relationships

## Contributing

When adding new features:
1. Follow the existing code structure
2. Add appropriate logging
3. Include validation where needed
4. Update this README

## License

This is a project template for educational and research purposes.
Always ensure you have permission to scrape websites and comply with their Terms of Service.
