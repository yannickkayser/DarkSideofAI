"""
Database utility functions using pure SQLite (no SQLAlchemy)
"""
from typing import Optional, List, Dict, Any
from config.database_sqlite import Database, init_database, check_database_status

from utils.logger import get_logger, log_database_operation

logger = get_logger(__name__)


class DatabaseManager:
    """
    Manager class for database operations using pure SQLite
    This is a wrapper around the Database class for compatibility
    """
    
    def __init__(self, db_path: str):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.db = Database(db_path)
    
    def __enter__(self):
        """Context manager entry"""
        self.db.connect()
        return self.db
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type is None:
            self.db.conn.commit()
        else:
            self.db.conn.rollback()
        self.db.close()


# Re-export for backwards compatibility
__all__ = [
    'DatabaseManager',
    'Database',
    'init_database',
    'check_database_status',
]
