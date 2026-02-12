"""
Main entry point for the web scraper
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import get_website_config, list_websites, DB_PATH
from database import Database
from scraper import WebScraper


def scrape_website(domain: str, max_pages: int = None):
    """
    Scrape a single website
    
    Args:
        domain: Domain name (e.g., 'mindrift.ai')
        max_pages: Maximum number of pages to scrape (None for unlimited)
    """
    # Get configuration
    config = get_website_config(domain)
    if not config:
        print(f"Error: No configuration found for {domain}")
        print(f"Available websites: {', '.join(list_websites())}")
        return
    
    # Initialize database
    db = Database(str(DB_PATH))
    
    # Create scraper and run
    scraper = WebScraper(config, db)
    stats = scraper.scrape_website(max_pages=max_pages)
    
    # Show database stats
    db.print_stats()
    
    # Close database
    db.close()


def scrape_all_websites(max_pages: int = None):
    """
    Scrape all configured websites
    
    Args:
        max_pages: Maximum pages per website
    """
    websites = list_websites()
    
    print(f"\nScraping {len(websites)} websites...\n")
    
    for domain in websites:
        print(f"\n{'='*60}")
        print(f"Starting: {domain}")
        print('='*60 + "\n")
        
        try:
            scrape_website(domain, max_pages)
        except Exception as e:
            print(f"\nError scraping {domain}: {e}\n")
            continue


def show_database_stats():
    """Show current database statistics"""
    db = Database(str(DB_PATH))
    db.print_stats()
    db.close()


def interactive_menu():
    """Interactive menu"""
    while True:
        print("\n" + "="*60)
        print("WEB SCRAPER - MAIN MENU")
        print("="*60)
        print("1. Scrape a single website")
        print("2. Scrape all websites")
        print("3. Show database statistics")
        print("4. List configured websites")
        print("5. Exit")
        print("="*60)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            print("\nAvailable websites:")
            for i, domain in enumerate(list_websites(), 1):
                print(f"  {i}. {domain}")
            
            domain = input("\nEnter domain name: ").strip()
            max_pages_input = input("Max pages (press Enter for unlimited): ").strip()
            max_pages = int(max_pages_input) if max_pages_input else None
            
            scrape_website(domain, max_pages)
        
        elif choice == '2':
            max_pages_input = input("Max pages per website (press Enter for unlimited): ").strip()
            max_pages = int(max_pages_input) if max_pages_input else None
            
            scrape_all_websites(max_pages)
        
        elif choice == '3':
            show_database_stats()
        
        elif choice == '4':
            print("\nConfigured websites:")
            print("-" * 60)
            for domain in list_websites():
                config = get_website_config(domain)
                print(f"\n{domain}")
                print(f"  Name: {config['name']}")
                print(f"  URL: {config['base_url']}")
                print(f"  Type: {config.get('type', 'Unknown')}")
        
        elif choice == '5':
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == '--all':
            max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else None
            scrape_all_websites(max_pages)
        elif sys.argv[1] == '--stats':
            show_database_stats()
        else:
            # Assume it's a domain name
            domain = sys.argv[1]
            max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else None
            scrape_website(domain, max_pages)
    else:
        # Interactive mode
        interactive_menu()
