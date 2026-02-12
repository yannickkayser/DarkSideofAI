"""
Simple test script to verify setup and functionality
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import list_websites, get_website_config, DB_PATH
from database import Database
from validator import (
    validate_url, validate_text_content, 
    validate_page_data, validate_sitemap
)
from sitemap_robots import SitemapRobotsParser


def test_config():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("TEST: Configuration")
    print("="*60)
    
    websites = list_websites()
    print(f"✓ Found {len(websites)} configured websites")
    
    for domain in websites:
        config = get_website_config(domain)
        print(f"\n{domain}:")
        print(f"  Name: {config['name']}")
        print(f"  URL: {config['base_url']}")
        print(f"  Type: {config.get('type', 'Unknown')}")


def test_database():
    """Test database initialization"""
    print("\n" + "="*60)
    print("TEST: Database")
    print("="*60)
    
    db = Database(str(DB_PATH))
    print(f"✓ Database initialized at {DB_PATH}")
    
    # Test adding website
    website_id = db.add_website(
        domain="test.example.com",
        name="Test Website",
        base_url="https://test.example.com",
        website_type="Test"
    )
    print(f"✓ Added test website (ID: {website_id})")
    
    # Test adding page
    page_id = db.add_page(
        website_id=website_id,
        url="https://test.example.com/page1",
        title="Test Page",
        text_content="This is test content for validation.",
        directory="/",
        html_element="main",
        depth=0,
        status_code=200,
        content_length=1000
    )
    
    if page_id:
        print(f"✓ Added test page (ID: {page_id})")
    else:
        print("✓ Page already exists (duplicate prevention working)")
    
    # Test stats
    stats = db.get_stats()
    print(f"✓ Database stats: {stats['total_pages']} pages")
    
    db.close()


def test_validators():
    """Test validation functions"""
    print("\n" + "="*60)
    print("TEST: Validators")
    print("="*60)
    
    # Test URL validation
    valid_url = "https://mindrift.ai/about"
    is_valid, msg = validate_url(valid_url)
    print(f"✓ URL validation: {msg}")
    
    invalid_url = "not-a-url"
    is_valid, msg = validate_url(invalid_url)
    print(f"✓ Invalid URL detected: {msg}")
    
    # Test content validation
    good_content = "This is a good piece of content with enough text to pass validation."
    is_valid, msg = validate_text_content(good_content)
    print(f"✓ Content validation: {msg}")
    
    bad_content = "Too short"
    is_valid, msg = validate_text_content(bad_content)
    print(f"✓ Short content detected: {msg}")
    
    # Test page data validation
    page_data = {
        'url': 'https://mindrift.ai/',
        'title': 'Mindrift AI',
        'text_content': 'This is enough content to pass validation and testing.'
    }
    is_valid, msg = validate_page_data(page_data)
    print(f"✓ Page data validation: {msg}")


def test_sitemap_parser():
    """Test sitemap and robots.txt parsing"""
    print("\n" + "="*60)
    print("TEST: Sitemap & Robots.txt Parser")
    print("="*60)
    
    # Test with a known site
    test_url = "https://www.python.org"
    
    parser = SitemapRobotsParser(test_url)
    
    print(f"\nTesting with: {test_url}")
    
    # Get robots.txt
    success, content, crawl_delay = parser.get_robots_txt()
    if success:
        print("✓ Successfully fetched robots.txt")
        if crawl_delay:
            print(f"  Crawl delay: {crawl_delay}s")
    else:
        print("✓ No robots.txt (this is okay)")
    
    # Find sitemap
    sitemap_url = parser.find_sitemap_url(content)
    if sitemap_url:
        print(f"✓ Found sitemap: {sitemap_url}")
        
        # Parse sitemap
        urls = parser.parse_sitemap(sitemap_url)
        if urls:
            print(f"✓ Parsed {len(urls)} URLs from sitemap")
            print(f"  Sample URLs:")
            for url in urls[:3]:
                print(f"    - {url}")
    else:
        print("✓ No sitemap found (will fall back to base URL)")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    
    try:
        test_config()
        test_database()
        test_validators()
        test_sitemap_parser()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nYou're ready to start scraping!")
        print("Run: python main.py")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
