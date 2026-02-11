"""
Test/Demo script for the scraping project
Run this to test the basic functionality
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import init_database, DB_PATH, get_scraper_config
from scrapers import StaticScraper
from utils import (
    validate_url, 
    normalize_url,
    check_validation_summary,
    DatabaseManager,
    FileHandler
)


def test_validation():
    """Test validation functions"""
    print("\n" + "="*60)
    print("TESTING VALIDATION FUNCTIONS")
    print("="*60)
    
    test_urls = [
        "https://mindrift.ai/",
        "http://example.com/page",
        "not a url",
        "ftp://invalid.com",
        "https://mindrift.ai/products/feature",
    ]
    
    validations = []
    for url in test_urls:
        result = validate_url(url, allowed_domains=['mindrift.ai', 'example.com'])
        validations.append(result)
        print(f"\nURL: {url}")
        print(f"Valid: {result['is_valid']}")
        if not result['is_valid']:
            print(f"Errors: {', '.join(result['errors'])}")
    
    check_validation_summary(validations, "URL")


def test_database():
    """Test database operations"""
    print("\n" + "="*60)
    print("TESTING DATABASE OPERATIONS")
    print("="*60)
    
    # Initialize database
    print("\nInitializing database...")
    init_database(DB_PATH)
    print("✓ Database initialized")
    
    # Test database operations
    with DatabaseManager(str(DB_PATH)) as db:
        # Create a test website
        website, created = db.get_or_create_website(
            domain="test.example.com",
            name="Test Website",
            base_url="https://test.example.com",
            scraper_type="static"
        )
        
        print(f"\n{'Created' if created else 'Retrieved'} website: {website['name']}")
        print(f"Website ID: {website['id']}")
        
        # Create a scrape session
        session_id = db.create_scrape_session(
            website_id=website['id'],
            config={"test": True}
        )
        print(f"Created scrape session: {session_id}")
        
        # Create a test page
        page_id = db.create_page(
            website_id=website['id'],
            scrape_session_id=session_id,
            url="https://test.example.com/page1",
            title="Test Page",
            text_content="This is test content",
            status_code=200,
            content_length=1000,
            is_successful=1
        )
        print(f"Created page: {page_id}")
        
        # Get statistics
        stats = db.get_scraping_stats(website['id'])
        print(f"\nStatistics:")
        print(f"  Total pages: {stats['total_pages']}")
        print(f"  Successful: {stats['successful']}")
    
    print("\n✓ Database operations completed successfully")


def test_file_handler():
    """Test file handling operations"""
    print("\n" + "="*60)
    print("TESTING FILE HANDLER")
    print("="*60)
    
    file_handler = FileHandler()
    
    # Create test content
    test_content = {
        'url': 'https://test.example.com/page1',
        'title': 'Test Page',
        'text_content': 'This is a test page with some content.',
        'links': [
            {'url': 'https://test.example.com/page2', 'anchor_text': 'Link 1'},
            {'url': 'https://test.example.com/page3', 'anchor_text': 'Link 2'},
        ],
        'images': [],
        'scraped_at': '2024-01-01T00:00:00'
    }
    
    # Save content
    print("\nSaving test content...")
    file_path = file_handler.save_page_content(
        url=test_content['url'],
        content_data=test_content,
        domain='test.example.com'
    )
    print(f"✓ Saved to: {file_path}")
    
    # Load content
    print("\nLoading content...")
    loaded_content = file_handler.load_page_content(file_path)
    print(f"✓ Loaded content for: {loaded_content['title']}")
    
    # Get stats
    print("\nFile statistics:")
    file_handler.print_domain_stats('test.example.com')


def test_scraper_config():
    """Test scraper configuration"""
    print("\n" + "="*60)
    print("TESTING SCRAPER CONFIGURATION")
    print("="*60)
    
    config = get_scraper_config("mindrift.ai")
    
    if config:
        print("\n✓ Configuration loaded successfully")
        print(f"Name: {config['name']}")
        print(f"Type: {config['scraper_type']}")
        print(f"Base URL: {config['base_url']}")
        print(f"Max Depth: {config['max_depth']}")
        print(f"Selectors defined: {len(config['selectors'])}")
    else:
        print("\n✗ Configuration not found")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    
    try:
        test_validation()
        test_scraper_config()
        test_database()
        test_file_handler()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test scraping project')
    parser.add_argument('--validation', action='store_true', help='Test validation')
    parser.add_argument('--database', action='store_true', help='Test database')
    parser.add_argument('--files', action='store_true', help='Test file handler')
    parser.add_argument('--config', action='store_true', help='Test configuration')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    if args.validation:
        test_validation()
    elif args.database:
        test_database()
    elif args.files:
        test_file_handler()
    elif args.config:
        test_scraper_config()
    elif args.all or len(sys.argv) == 1:
        run_all_tests()
    else:
        parser.print_help()
