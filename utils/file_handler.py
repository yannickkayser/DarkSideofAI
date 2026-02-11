"""
File handling utilities for saving scraped content
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlparse
import requests

from config.settings import RAW_DATA_DIR, MEDIA_DIR, MAX_MEDIA_SIZE
from utils.logger import get_logger
from utils.validation import sanitize_filename

logger = get_logger(__name__)


class FileHandler:
    """Handler for file operations related to scraped content"""
    
    def __init__(self, raw_dir: Path = RAW_DATA_DIR, media_dir: Path = MEDIA_DIR):
        """
        Initialize file handler
        
        Args:
            raw_dir: Directory for raw JSON data
            media_dir: Directory for media files
        """
        self.raw_dir = raw_dir
        self.media_dir = media_dir
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)
    
    def save_page_content(self, url: str, content_data: Dict[str, Any], 
                         domain: str) -> str:
        """
        Save page content to JSON file
        
        Args:
            url: Page URL
            content_data: Dictionary containing all page data
            domain: Website domain
        
        Returns:
            str: Relative path to saved file
        """
        # Create domain-specific directory
        domain_dir = self.raw_dir / sanitize_filename(domain)
        domain_dir.mkdir(exist_ok=True)
        
        # Generate filename from URL hash
        url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"{url_hash}_{timestamp}.json"
        
        file_path = domain_dir / filename
        
        # Add metadata
        content_data['_metadata'] = {
            'url': url,
            'saved_at': datetime.utcnow().isoformat(),
            'file_path': str(file_path.relative_to(self.raw_dir))
        }
        
        # Save to JSON
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(content_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved content to {file_path}")
            return str(file_path.relative_to(self.raw_dir))
            
        except Exception as e:
            logger.error(f"Error saving content to {file_path}: {e}")
            raise
    
    def load_page_content(self, relative_path: str) -> Dict[str, Any]:
        """
        Load page content from JSON file
        
        Args:
            relative_path: Relative path to JSON file
        
        Returns:
            Dictionary with page content
        """
        file_path = self.raw_dir / relative_path
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading content from {file_path}: {e}")
            raise
    
    def download_media(self, url: str, media_type: str, domain: str, 
                      alt_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Download media file (image, video, etc.)
        
        Args:
            url: Media URL
            media_type: Type of media (image, video, pdf)
            domain: Website domain
            alt_text: Alternative text for naming
        
        Returns:
            Dictionary with download info or None if failed
        """
        try:
            # Create domain-specific media directory
            domain_media_dir = self.media_dir / sanitize_filename(domain) / media_type
            domain_media_dir.mkdir(parents=True, exist_ok=True)
            
            # Download with size limit
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Check content length
            content_length = int(response.headers.get('content-length', 0))
            if content_length > MAX_MEDIA_SIZE:
                logger.warning(f"Media too large ({content_length} bytes): {url}")
                return None
            
            # Generate filename
            parsed_url = urlparse(url)
            original_filename = Path(parsed_url.path).name or 'unnamed'
            
            # Use alt text for better naming if available
            if alt_text:
                safe_alt = sanitize_filename(alt_text[:50])
                extension = Path(original_filename).suffix or '.jpg'
                filename = f"{safe_alt}{extension}"
            else:
                filename = original_filename
            
            # Ensure unique filename
            file_path = domain_media_dir / sanitize_filename(filename)
            counter = 1
            while file_path.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                file_path = domain_media_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Download and save
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Get file info
            file_size = file_path.stat().st_size
            mime_type = response.headers.get('content-type', '')
            
            logger.info(f"Downloaded media: {url} -> {file_path}")
            
            return {
                'file_path': str(file_path.relative_to(self.media_dir)),
                'file_size': file_size,
                'mime_type': mime_type,
                'original_url': url
            }
            
        except Exception as e:
            logger.error(f"Error downloading media from {url}: {e}")
            return None
    
    def save_batch_content(self, pages_data: list, domain: str) -> list:
        """
        Save multiple pages at once
        
        Args:
            pages_data: List of page content dictionaries
            domain: Website domain
        
        Returns:
            List of saved file paths
        """
        saved_paths = []
        
        for page_data in pages_data:
            try:
                url = page_data.get('url', 'unknown')
                path = self.save_page_content(url, page_data, domain)
                saved_paths.append(path)
            except Exception as e:
                logger.error(f"Error in batch save for {url}: {e}")
                saved_paths.append(None)
        
        logger.info(f"Batch saved {len([p for p in saved_paths if p])} of {len(pages_data)} pages")
        return saved_paths
    
    def create_content_index(self, domain: str) -> Dict[str, Any]:
        """
        Create an index of all saved content for a domain
        
        Args:
            domain: Website domain
        
        Returns:
            Dictionary with index information
        """
        domain_dir = self.raw_dir / sanitize_filename(domain)
        
        if not domain_dir.exists():
            return {'error': 'Domain directory not found'}
        
        index = {
            'domain': domain,
            'created_at': datetime.utcnow().isoformat(),
            'total_files': 0,
            'files': []
        }
        
        for json_file in domain_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                index['files'].append({
                    'filename': json_file.name,
                    'url': data.get('url', 'unknown'),
                    'title': data.get('title', 'unknown'),
                    'saved_at': data.get('_metadata', {}).get('saved_at'),
                    'size': json_file.stat().st_size
                })
                
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
        
        index['total_files'] = len(index['files'])
        
        # Save index
        index_path = domain_dir / '_index.json'
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"Created index for {domain}: {index['total_files']} files")
        return index
    
    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """
        Get statistics for saved content of a domain
        
        Args:
            domain: Website domain
        
        Returns:
            Dictionary with statistics
        """
        domain_dir = self.raw_dir / sanitize_filename(domain)
        
        if not domain_dir.exists():
            return {'error': 'Domain directory not found'}
        
        json_files = list(domain_dir.glob('*.json'))
        total_size = sum(f.stat().st_size for f in json_files if f.name != '_index.json')
        
        stats = {
            'domain': domain,
            'total_files': len(json_files) - (1 if (domain_dir / '_index.json').exists() else 0),
            'total_size_mb': total_size / 1024 / 1024,
            'directory': str(domain_dir)
        }
        
        # Media stats
        media_domain_dir = self.media_dir / sanitize_filename(domain)
        if media_domain_dir.exists():
            media_files = list(media_domain_dir.rglob('*'))
            media_files = [f for f in media_files if f.is_file()]
            media_size = sum(f.stat().st_size for f in media_files)
            
            stats['media_files'] = len(media_files)
            stats['media_size_mb'] = media_size / 1024 / 1024
        else:
            stats['media_files'] = 0
            stats['media_size_mb'] = 0
        
        return stats
    
    def print_domain_stats(self, domain: str) -> None:
        """Print domain statistics - feedback function"""
        stats = self.get_domain_stats(domain)
        
        if 'error' in stats:
            print(f"\nError: {stats['error']}\n")
            return
        
        print("\n" + "="*60)
        print(f"FILE STATISTICS - {domain}")
        print("="*60)
        print(f"Total JSON Files: {stats['total_files']}")
        print(f"Total Size: {stats['total_size_mb']:.2f} MB")
        print(f"Media Files: {stats['media_files']}")
        print(f"Media Size: {stats['media_size_mb']:.2f} MB")
        print(f"Directory: {stats['directory']}")
        print("="*60 + "\n")


def check_file_structure(base_dir: Path = RAW_DATA_DIR) -> None:
    """
    Check and display file structure - feedback function
    
    Args:
        base_dir: Base directory to check
    """
    print("\n" + "="*60)
    print("FILE STRUCTURE")
    print("="*60)
    
    if not base_dir.exists():
        print("Directory does not exist yet")
        print("="*60 + "\n")
        return
    
    for domain_dir in base_dir.iterdir():
        if domain_dir.is_dir():
            json_files = list(domain_dir.glob('*.json'))
            total_size = sum(f.stat().st_size for f in json_files)
            
            print(f"\n{domain_dir.name}/")
            print(f"  Files: {len(json_files)}")
            print(f"  Size: {total_size / 1024:.2f} KB")
    
    print("\n" + "="*60 + "\n")
