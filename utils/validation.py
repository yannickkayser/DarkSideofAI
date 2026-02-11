"""
Validation utilities for scraping project
Provides functions to validate URLs, content, and data integrity
"""
import re
from urllib.parse import urlparse, urljoin
from typing import Optional, List, Dict, Any
from utils.logger import get_logger, log_validation_error

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_url(url: str, allowed_domains: Optional[List[str]] = None) -> dict:
    """
    Validate a URL and return validation results
    
    Args:
        url: URL to validate
        allowed_domains: Optional list of allowed domains
    
    Returns:
        dict: {
            'is_valid': bool,
            'url': str (cleaned),
            'scheme': str,
            'domain': str,
            'errors': list of error messages
        }
    """
    result = {
        'is_valid': False,
        'url': url,
        'scheme': None,
        'domain': None,
        'errors': []
    }
    
    if not url or not isinstance(url, str):
        result['errors'].append("URL is empty or not a string")
        return result
    
    # Remove whitespace
    url = url.strip()
    result['url'] = url
    
    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        result['errors'].append(f"URL parsing failed: {str(e)}")
        return result
    
    # Validate scheme
    if parsed.scheme not in ['http', 'https']:
        result['errors'].append(f"Invalid scheme: {parsed.scheme} (expected http/https)")
        return result
    result['scheme'] = parsed.scheme
    
    # Validate domain
    if not parsed.netloc:
        result['errors'].append("No domain found in URL")
        return result
    result['domain'] = parsed.netloc
    
    # Check against allowed domains
    if allowed_domains:
        domain_match = False
        for allowed in allowed_domains:
            if parsed.netloc == allowed or parsed.netloc.endswith(f".{allowed}"):
                domain_match = True
                break
        
        if not domain_match:
            result['errors'].append(f"Domain {parsed.netloc} not in allowed list")
            return result
    
    # If we got here, URL is valid
    result['is_valid'] = True
    return result


def is_valid_url(url: str, allowed_domains: Optional[List[str]] = None) -> bool:
    """
    Quick check if URL is valid
    
    Args:
        url: URL to check
        allowed_domains: Optional list of allowed domains
    
    Returns:
        bool: True if valid, False otherwise
    """
    return validate_url(url, allowed_domains)['is_valid']


def normalize_url(url: str, base_url: Optional[str] = None) -> str:
    """
    Normalize a URL (resolve relative URLs, remove fragments, etc.)
    
    Args:
        url: URL to normalize
        base_url: Base URL for resolving relative URLs
    
    Returns:
        str: Normalized URL
    """
    if not url:
        return ""
    
    url = url.strip()
    
    # Resolve relative URLs
    if base_url and not url.startswith(('http://', 'https://')):
        url = urljoin(base_url, url)
    
    # Parse and reconstruct without fragment
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    
    # Add query string if present
    if parsed.query:
        normalized += f"?{parsed.query}"
    
    # Remove trailing slash from path (except for root)
    if normalized.endswith('/') and parsed.path != '/':
        normalized = normalized[:-1]
    
    return normalized


def validate_content(content: str, min_length: int = 100) -> dict:
    """
    Validate scraped content
    
    Args:
        content: Text content to validate
        min_length: Minimum acceptable content length
    
    Returns:
        dict: {
            'is_valid': bool,
            'length': int,
            'warnings': list of warning messages,
            'errors': list of error messages
        }
    """
    result = {
        'is_valid': False,
        'length': 0,
        'warnings': [],
        'errors': []
    }
    
    if not content:
        result['errors'].append("Content is empty")
        return result
    
    if not isinstance(content, str):
        result['errors'].append("Content is not a string")
        return result
    
    # Check length
    content_length = len(content.strip())
    result['length'] = content_length
    
    if content_length < min_length:
        result['warnings'].append(f"Content length ({content_length}) below minimum ({min_length})")
    
    # Check for common error patterns
    error_indicators = [
        '404', 'not found', 'error', 'access denied',
        'forbidden', 'unauthorized', 'page not found'
    ]
    
    content_lower = content.lower()
    for indicator in error_indicators:
        if indicator in content_lower and content_length < 500:
            result['warnings'].append(f"Content may contain error page (found '{indicator}')")
            break
    
    # If no critical errors, mark as valid
    if not result['errors']:
        result['is_valid'] = True
    
    return result


def validate_page_data(page_data: Dict[str, Any], required_fields: Optional[List[str]] = None) -> dict:
    """
    Validate a page data dictionary
    
    Args:
        page_data: Dictionary containing page data
        required_fields: List of required field names
    
    Returns:
        dict: {
            'is_valid': bool,
            'missing_fields': list,
            'invalid_fields': dict,
            'errors': list
        }
    """
    result = {
        'is_valid': False,
        'missing_fields': [],
        'invalid_fields': {},
        'errors': []
    }
    
    if not isinstance(page_data, dict):
        result['errors'].append("page_data is not a dictionary")
        return result
    
    # Default required fields
    if required_fields is None:
        required_fields = ['url', 'title', 'content']
    
    # Check for missing fields
    for field in required_fields:
        if field not in page_data or page_data[field] is None:
            result['missing_fields'].append(field)
    
    # Validate specific fields
    if 'url' in page_data:
        url_validation = validate_url(page_data['url'])
        if not url_validation['is_valid']:
            result['invalid_fields']['url'] = url_validation['errors']
    
    if 'status_code' in page_data:
        status = page_data['status_code']
        if not isinstance(status, int) or status < 100 or status >= 600:
            result['invalid_fields']['status_code'] = f"Invalid HTTP status code: {status}"
    
    if 'content' in page_data and page_data['content']:
        content_validation = validate_content(page_data['content'])
        if not content_validation['is_valid']:
            result['invalid_fields']['content'] = content_validation['errors']
    
    # Mark as valid if no critical issues
    if not result['missing_fields'] and not result['invalid_fields'] and not result['errors']:
        result['is_valid'] = True
    
    return result


def matches_pattern(url: str, patterns: List[str]) -> bool:
    """
    Check if URL matches any of the given regex patterns
    
    Args:
        url: URL to check
        patterns: List of regex patterns
    
    Returns:
        bool: True if matches any pattern
    """
    if not patterns:
        return False
    
    for pattern in patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False


def should_scrape_url(url: str, include_patterns: List[str], exclude_patterns: List[str]) -> bool:
    """
    Determine if a URL should be scraped based on include/exclude patterns
    
    Args:
        url: URL to check
        include_patterns: List of patterns to include
        exclude_patterns: List of patterns to exclude
    
    Returns:
        bool: True if should scrape, False otherwise
    """
    # If exclude patterns match, skip
    if exclude_patterns and matches_pattern(url, exclude_patterns):
        return False
    
    # If no include patterns, scrape by default
    if not include_patterns:
        return True
    
    # Only scrape if matches include patterns
    return matches_pattern(url, include_patterns)


def validate_file_path(file_path: str, must_exist: bool = False) -> dict:
    """
    Validate a file path
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must already exist
    
    Returns:
        dict: Validation results
    """
    from pathlib import Path
    
    result = {
        'is_valid': False,
        'path': file_path,
        'exists': False,
        'is_file': False,
        'errors': []
    }
    
    try:
        path = Path(file_path)
        result['exists'] = path.exists()
        result['is_file'] = path.is_file()
        
        if must_exist and not result['exists']:
            result['errors'].append("File does not exist")
            return result
        
        # Check if parent directory exists
        if not path.parent.exists():
            result['errors'].append("Parent directory does not exist")
            return result
        
        result['is_valid'] = True
        
    except Exception as e:
        result['errors'].append(f"Path validation error: {str(e)}")
    
    return result


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename by removing invalid characters
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
    
    Returns:
        str: Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    
    # Truncate if too long
    if len(sanitized) > max_length:
        # Keep file extension
        parts = sanitized.rsplit('.', 1)
        if len(parts) == 2:
            name, ext = parts
            max_name_length = max_length - len(ext) - 1
            sanitized = f"{name[:max_name_length]}.{ext}"
        else:
            sanitized = sanitized[:max_length]
    
    return sanitized or 'unnamed'


# Feedback/check functions for user
def check_validation_summary(validations: List[dict], item_type: str = "item") -> None:
    """
    Print a summary of validation results
    
    Args:
        validations: List of validation result dictionaries
        item_type: Type of items being validated (for display)
    """
    total = len(validations)
    valid = sum(1 for v in validations if v.get('is_valid', False))
    invalid = total - valid
    
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY - {item_type.upper()}")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"Valid: {valid} ({valid/total*100:.1f}%)" if total > 0 else "Valid: 0")
    print(f"Invalid: {invalid} ({invalid/total*100:.1f}%)" if total > 0 else "Invalid: 0")
    
    if invalid > 0:
        print(f"\n{'-'*60}")
        print("VALIDATION ERRORS:")
        print(f"{'-'*60}")
        for i, v in enumerate(validations):
            if not v.get('is_valid', False):
                errors = v.get('errors', []) + v.get('warnings', [])
                print(f"\n{i+1}. {item_type.capitalize()}: {v.get('url', v.get('path', 'unknown'))}")
                for error in errors:
                    print(f"   - {error}")
    
    print(f"{'='*60}\n")
