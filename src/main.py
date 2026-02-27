"""
Main entry point for the web scraper
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import get_website_config, list_websites, DB_PATH
from database import Database
from scraper import WebScraper
from sitemap_robots import SitemapRobotsParser
from crawler import LinkCrawler


def scrape_website(domain: str, max_pages: int = None, use_crawler: bool = False):
    """
    Scrape a single website

    Args:
        domain: Domain name (e.g., 'crowdgen.com')
        max_pages: Maximum number of pages to scrape (None for unlimited)
        use_crawler: Use BFS link crawler instead of sitemap discovery
    """
    config = get_website_config(domain)
    if not config:
        print(f"Error: No configuration found for {domain}")
        print(f"Available websites: {', '.join(list_websites())}")
        return

    db = Database(str(DB_PATH))
    scraper = WebScraper(config, db)

    if use_crawler:
        # Discover URLs via BFS crawl, then inject into scraper
        crawler = LinkCrawler(
            base_url=config["base_url"],
            max_pages=max_pages or 500,
            max_depth=config.get("max_depth", 3),
            rate_limit=config.get("rate_limit", 1.5),
        )
        urls = crawler.discover_urls()

        if not urls:
            print("No URLs discovered — aborting.")
            db.close()
            return

        # Patch sitemap step to use crawled URLs instead
        original_get_all = SitemapRobotsParser.get_all_urls

        def patched_get_all(self_inner):
            print(f"[crawler] Injecting {len(urls)} crawled URLs into scraper")
            return urls, None

        SitemapRobotsParser.get_all_urls = patched_get_all

        try:
            stats = scraper.scrape_website(max_pages=max_pages)
        finally:
            SitemapRobotsParser.get_all_urls = original_get_all
    else:
        stats = scraper.scrape_website(max_pages=max_pages)

    db.print_stats()
    db.close()


def scrape_all_websites(max_pages: int = None):
    """Scrape all configured websites using sitemap discovery"""
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
        print("1. Scrape a single website (sitemap)")
        print("2. Scrape a single website (crawler - no sitemap)")
        print("3. Scrape all websites")
        print("4. Show database statistics")
        print("5. List configured websites")
        print("6. Exit")
        print("="*60)

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice in ('1', '2'):
            print("\nAvailable websites:")
            for i, domain in enumerate(list_websites(), 1):
                print(f"  {i}. {domain}")

            domain = input("\nEnter domain name: ").strip()
            max_pages_input = input("Max pages (press Enter for unlimited): ").strip()
            max_pages = int(max_pages_input) if max_pages_input else None

            scrape_website(domain, max_pages, use_crawler=(choice == '2'))

        elif choice == '3':
            max_pages_input = input("Max pages per website (press Enter for unlimited): ").strip()
            max_pages = int(max_pages_input) if max_pages_input else None
            scrape_all_websites(max_pages)

        elif choice == '4':
            show_database_stats()

        elif choice == '5':
            print("\nConfigured websites:")
            print("-" * 60)
            for domain in list_websites():
                config = get_website_config(domain)
                print(f"\n{domain}")
                print(f"  Name: {config['name']}")
                print(f"  URL: {config['base_url']}")
                print(f"  Type: {config.get('type', 'Unknown')}")

        elif choice == '6':
            print("\nGoodbye!")
            break

        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == '--all':
            max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else None
            scrape_all_websites(max_pages)
        elif sys.argv[1] == '--stats':
            show_database_stats()
        elif sys.argv[1] == '--crawl':
            # e.g. python main.py --crawl crowdgen.com 200
            domain = sys.argv[2] if len(sys.argv) > 2 else None
            if not domain:
                print("Usage: python main.py --crawl <domain> [max_pages]")
                sys.exit(1)
            max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else None
            scrape_website(domain, max_pages, use_crawler=True)
        else:
            # Assume it's a domain name, use sitemap by default
            domain = sys.argv[1]
            max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else None
            scrape_website(domain, max_pages)
    else:
        interactive_menu()
