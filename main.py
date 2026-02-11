"""
Main entry point for the web scraping project
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    get_scraper_config, list_configured_sites,
    init_database, DB_PATH, DB_ECHO
)
from scrapers.static_scraper import StaticScraper
from utils import (
    get_logger,
    check_database_status,
    check_file_structure,
    DatabaseManager,
    FileHandler
)

logger = get_logger(__name__)


def initialize_project():
    """Initialize the project (database, directories, etc.)"""
    print("Initializing scraping project...")
    
    # Initialize database
    engine, SessionLocal = init_database(DB_PATH, echo=DB_ECHO)
    print(f"✓ Database initialized at {DB_PATH}")
    
    # Check directories
    from config.settings import DATA_DIR, RAW_DATA_DIR, MEDIA_DIR, LOGS_DIR
    for directory in [DATA_DIR, RAW_DATA_DIR, MEDIA_DIR, LOGS_DIR]:
        if directory.exists():
            print(f"✓ Directory exists: {directory}")
        else:
            print(f"✗ Directory missing: {directory}")
    
    print("\nProject initialized successfully!\n")


def scrape_site(domain: str, max_pages: int = None):
    """
    Scrape a configured website
    
    Args:
        domain: Domain name (e.g., "mindrift.ai")
        max_pages: Optional maximum number of pages to scrape
    """
    # Get configuration
    config = get_scraper_config(domain)
    
    if not config:
        print(f"Error: No configuration found for {domain}")
        print(f"Available sites: {', '.join(list_configured_sites())}")
        return
    
    print(f"\n{'='*60}")
    print(f"Scraping: {config['name']} ({domain})")
    print(f"{'='*60}\n")
    
    # Create appropriate scraper
    scraper_type = config.get('scraper_type', 'static')
    
    if scraper_type == 'static':
        scraper = StaticScraper(config, str(DB_PATH))
    else:
        # For now, fall back to static
        # We'll create DynamicScraper next if needed
        print(f"Warning: {scraper_type} scraper not yet implemented, using static")
        scraper = StaticScraper(config, str(DB_PATH))
    
    # Start scraping
    try:
        result = scraper.scrape_website(max_pages=max_pages)
        
        print("\n✓ Scraping completed successfully!")
        print(f"Website ID: {result['website_id']}")
        print(f"Session ID: {result['session_id']}")
        
        # Show file statistics
        file_handler = FileHandler()
        file_handler.print_domain_stats(domain)
        
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        print(f"\n✗ Scraping failed: {e}")


def show_status():
    """Show current database and file status"""
    print("\nChecking project status...\n")
    
    # Database status
    check_database_status(str(DB_PATH))
    
    # File structure
    check_file_structure()


def list_sites():
    """List all configured sites"""
    sites = list_configured_sites()
    
    print("\n" + "="*60)
    print("CONFIGURED WEBSITES")
    print("="*60)
    
    for domain in sites:
        config = get_scraper_config(domain)
        print(f"\n{domain}")
        print(f"  Name: {config['name']}")
        print(f"  Type: {config['scraper_type']}")
        print(f"  URL: {config['base_url']}")
        print(f"  Max Depth: {config['max_depth']}")
    
    print("\n" + "="*60 + "\n")


def interactive_menu():
    """Interactive menu for the scraping project"""
    while True:
        print("\n" + "="*60)
        print("WEB SCRAPING PROJECT - MAIN MENU")
        print("="*60)
        print("1. Initialize project")
        print("2. Scrape a website")
        print("3. Show project status")
        print("4. List configured websites")
        print("5. Exit")
        print("="*60)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            initialize_project()
        
        elif choice == '2':
            list_sites()
            domain = input("\nEnter domain to scrape: ").strip()
            max_pages_input = input("Max pages (press Enter for unlimited): ").strip()
            max_pages = int(max_pages_input) if max_pages_input else None
            scrape_site(domain, max_pages)
        
        elif choice == '3':
            show_status()
        
        elif choice == '4':
            list_sites()
        
        elif choice == '5':
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Web Scraping Project')
    parser.add_argument('--init', action='store_true', help='Initialize the project')
    parser.add_argument('--scrape', type=str, help='Domain to scrape')
    parser.add_argument('--max-pages', type=int, help='Maximum pages to scrape')
    parser.add_argument('--status', action='store_true', help='Show project status')
    parser.add_argument('--list', action='store_true', help='List configured sites')
    parser.add_argument('--interactive', action='store_true', help='Run interactive menu')
    
    args = parser.parse_args()
    
    # Handle command line arguments
    if args.init:
        initialize_project()
    elif args.scrape:
        scrape_site(args.scrape, args.max_pages)
    elif args.status:
        show_status()
    elif args.list:
        list_sites()
    elif args.interactive or len(sys.argv) == 1:
        # Interactive mode if no args or --interactive flag
        interactive_menu()
    else:
        parser.print_help()
