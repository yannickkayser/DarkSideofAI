"""
Demo script for sitemap and robots.txt functionality
Shows how to use sitemaps to efficiently scrape websites
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.sitemap_parser import (
    check_robots_and_sitemap,
    SitemapBasedScraper,
    RobotsParser,
    SitemapParser
)


def demo_robots_check(url: str):
    """
    Demo: Check robots.txt for a website
    
    Args:
        url: Website URL
    """
    print("\n" + "="*60)
    print(f"CHECKING ROBOTS.TXT FOR: {url}")
    print("="*60)
    
    robots = RobotsParser(url)
    success = robots.fetch_and_parse()
    
    if success:
        robots.print_info()
        
        # Test some URLs
        test_urls = [
            f"{url}/",
            f"{url}/about",
            f"{url}/api",
            f"{url}/admin",
        ]
        
        print("URL Permission Tests:")
        print("-" * 60)
        for test_url in test_urls:
            can_fetch = robots.can_fetch(test_url)
            status = "✓ ALLOWED" if can_fetch else "✗ DISALLOWED"
            print(f"{status}: {test_url}")
    else:
        print("Could not fetch robots.txt")


def demo_sitemap_check(url: str):
    """
    Demo: Find and parse sitemap for a website
    
    Args:
        url: Website URL
    """
    print("\n" + "="*60)
    print(f"CHECKING SITEMAP FOR: {url}")
    print("="*60)
    
    # Get robots.txt first
    robots = RobotsParser(url)
    robots.fetch_and_parse()
    
    # Parse sitemap
    sitemap = SitemapParser(url)
    urls = sitemap.get_all_urls(robots.content)
    
    if urls:
        print(f"\n✓ Found {len(urls)} URLs in sitemap")
        
        # Show sample URLs
        print("\nSample URLs (first 10):")
        print("-" * 60)
        for i, url_data in enumerate(urls[:10], 1):
            print(f"{i}. {url_data['url']}")
            if 'priority' in url_data:
                print(f"   Priority: {url_data['priority']}")
            if 'lastmod' in url_data:
                print(f"   Last Modified: {url_data['lastmod']}")
        
        # Statistics
        print("\nStatistics:")
        print("-" * 60)
        
        # Priority distribution
        priorities = {}
        for url_data in urls:
            p = url_data.get('priority', 'unknown')
            priorities[p] = priorities.get(p, 0) + 1
        
        print("By Priority:")
        for priority, count in sorted(priorities.items(), reverse=True):
            print(f"  {priority}: {count} URLs")
        
        # Change frequency distribution
        changefreqs = {}
        for url_data in urls:
            cf = url_data.get('changefreq', 'unknown')
            changefreqs[cf] = changefreqs.get(cf, 0) + 1
        
        print("\nBy Change Frequency:")
        for freq, count in changefreqs.items():
            print(f"  {freq}: {count} URLs")
    else:
        print("\n✗ No sitemap found or no URLs extracted")


def demo_full_check(url: str):
    """
    Demo: Complete robots.txt and sitemap analysis
    
    Args:
        url: Website URL
    """
    result = check_robots_and_sitemap(url)
    
    print("\nComplete Analysis Summary:")
    print("="*60)
    print(f"Robots.txt exists: {result['robots_exists']}")
    print(f"Sitemaps found: {len(result['sitemap_urls'])}")
    print(f"Total URLs: {result['total_urls']}")
    if result['recommended_crawl_delay']:
        print(f"Recommended crawl delay: {result['recommended_crawl_delay']}s")


def demo_filtered_urls(url: str, pattern: str):
    """
    Demo: Get filtered URLs from sitemap
    
    Args:
        url: Website URL
        pattern: Regex pattern to filter URLs
    """
    print("\n" + "="*60)
    print(f"FILTERED URLS FROM: {url}")
    print(f"Pattern: {pattern}")
    print("="*60)
    
    scraper = SitemapBasedScraper(url)
    scraper.robots.fetch_and_parse()
    
    urls = scraper.get_allowed_urls(filter_patterns=[pattern])
    
    print(f"\n✓ Found {len(urls)} matching URLs")
    
    for i, url_data in enumerate(urls[:20], 1):
        print(f"{i}. {url_data['url']}")


def interactive_demo():
    """Interactive demo menu"""
    while True:
        print("\n" + "="*60)
        print("SITEMAP & ROBOTS.TXT DEMO")
        print("="*60)
        print("1. Check robots.txt")
        print("2. Check sitemap")
        print("3. Full analysis (robots.txt + sitemap)")
        print("4. Filter URLs by pattern")
        print("5. Test with common websites")
        print("6. Exit")
        print("="*60)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            url = input("Enter website URL: ").strip()
            demo_robots_check(url)
        
        elif choice == '2':
            url = input("Enter website URL: ").strip()
            demo_sitemap_check(url)
        
        elif choice == '3':
            url = input("Enter website URL: ").strip()
            demo_full_check(url)
        
        elif choice == '4':
            url = input("Enter website URL: ").strip()
            pattern = input("Enter regex pattern (e.g., '.*/blog/.*'): ").strip()
            demo_filtered_urls(url, pattern)
        
        elif choice == '5':
            test_sites = [
                "https://www.python.org",
                "https://docs.github.com",
                "https://stackoverflow.com",
            ]
            
            print("\nTesting common websites...")
            for site in test_sites:
                print(f"\n{'='*60}")
                print(f"Testing: {site}")
                print('='*60)
                try:
                    demo_full_check(site)
                except Exception as e:
                    print(f"Error: {e}")
        
        elif choice == '6':
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sitemap and robots.txt demo')
    parser.add_argument('--url', type=str, help='Website URL to check')
    parser.add_argument('--robots', action='store_true', help='Check robots.txt only')
    parser.add_argument('--sitemap', action='store_true', help='Check sitemap only')
    parser.add_argument('--full', action='store_true', help='Full analysis')
    parser.add_argument('--filter', type=str, help='Filter URLs by regex pattern')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.url:
        if args.robots:
            demo_robots_check(args.url)
        elif args.sitemap:
            demo_sitemap_check(args.url)
        elif args.filter:
            demo_filtered_urls(args.url, args.filter)
        else:
            demo_full_check(args.url)
    elif args.interactive or len(sys.argv) == 1:
        interactive_demo()
    else:
        parser.print_help()
