"""
Validation functions to check if scraping is working correctly
"""
from urllib.parse import urlparse
from typing import Tuple


def validate_url(url: str) -> Tuple[bool, str]:
    """
    Validate URL
    
    Returns:
        (is_valid, message)
    """
    if not url:
        return False, "URL is empty"
    
    try:
        parsed = urlparse(url)
        if not parsed.scheme in ['http', 'https']:
            return False, f"Invalid scheme: {parsed.scheme}"
        if not parsed.netloc:
            return False, "No domain found"
        return True, "Valid URL"
    except Exception as e:
        return False, f"URL parsing error: {e}"


def validate_text_content(text: str, min_length: int = 50) -> Tuple[bool, str]:
    """
    Validate extracted text content
    
    Returns:
        (is_valid, message)
    """
    if not text:
        return False, "No text content extracted"
    
    text = text.strip()
    
    if len(text) < min_length:
        return False, f"Content too short ({len(text)} chars, minimum {min_length})"
    
    return True, f"Valid content ({len(text)} chars)"


def validate_page_data(page_data: dict) -> Tuple[bool, str]:
    """
    Validate complete page data
    
    Returns:
        (is_valid, message)
    """
    required_fields = ['url', 'title', 'text_content']
    
    # Check required fields
    for field in required_fields:
        if field not in page_data or not page_data[field]:
            return False, f"Missing required field: {field}"
    
    # Validate URL
    url_valid, url_msg = validate_url(page_data['url'])
    if not url_valid:
        return False, url_msg
    
    # Validate content
    content_valid, content_msg = validate_text_content(page_data['text_content'])
    if not content_valid:
        return False, content_msg
    
    return True, "All validations passed"


def validate_sitemap(sitemap_urls: list) -> Tuple[bool, str]:
    """
    Validate sitemap data
    
    Returns:
        (is_valid, message)
    """
    if not sitemap_urls:
        return False, "No URLs found in sitemap"
    
    if not isinstance(sitemap_urls, list):
        return False, "Sitemap URLs must be a list"
    
    return True, f"Sitemap contains {len(sitemap_urls)} URLs"


def validate_robots_txt(robots_content: str) -> Tuple[bool, str]:
    """
    Validate robots.txt content
    
    Returns:
        (is_valid, message)
    """
    if not robots_content:
        return False, "No robots.txt content"
    
    # Check for basic robots.txt structure
    if 'user-agent' not in robots_content.lower():
        return False, "Invalid robots.txt format"
    
    return True, "Valid robots.txt"
