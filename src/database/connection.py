"""
src/database/connection.py
--------------------------
PostgreSQL connection and session management.
"""

import logging
import os
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

from src.database.models import Base

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Manages PostgreSQL database connection and session creation.
    
    Features:
    - Connection pooling
    - Automatic table creation
    - Session factory
    """
    
    def __init__(self, database_url: str = None, echo: bool = False):
        """
        Initialize database connection.
        
        Args:
            database_url: PostgreSQL connection URL
                Format: postgresql://user:password@host:port/dbname
                Default: Read from DB_URL environment variable
            echo: If True, log all SQL statements
        """
        # Build database URL
        if database_url is None:
            database_url = os.getenv(
                "DB_URL",
                os.getenv(
                    "DATABASE_URL",
                    "postgresql://postgres:password@localhost:5432/rag_db"
                )
            )
        
        self.database_url = database_url
        
        logger.info(f"Connecting to database: {self._mask_url(database_url)}")
        
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                database_url,
                echo=echo,
                pool_pre_ping=True,  # Test connections before using them
                poolclass=NullPool if os.getenv("DB_POOL_DISABLED") else None
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("✓ Database connection successful")
            
            # Create all tables
            self.create_tables()
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            raise
    
    def create_tables(self):
        """Create all tables if they don't exist."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("✓ Database tables initialized")
        except Exception as e:
            logger.error(f"✗ Failed to create tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def get_session_generator(self) -> Generator[Session, None, None]:
        """
        Session generator for dependency injection.
        Useful for FastAPI dependencies.
        """
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    @staticmethod
    def _mask_url(url: str) -> str:
        """Mask password in database URL for logging."""
        if "@" in url:
            scheme, rest = url.split("://", 1)
            user_pass, host = rest.split("@", 1)
            user = user_pass.split(":")[0]
            return f"{scheme}://{user}:***@{host}"
        return url
    
    def close(self):
        """Close all connections in the pool."""
        self.engine.dispose()
        logger.info("Database connection pool closed")


# Global database instance
_db_instance = None


def get_database(database_url: str = None) -> DatabaseConnection:
    """
    Get or create the global database connection.
    
    Args:
        database_url: PostgreSQL connection URL (optional)
        
    Returns:
        DatabaseConnection instance
    """
    global _db_instance
    
    if _db_instance is None:
        _db_instance = DatabaseConnection(database_url=database_url)
    
    return _db_instance


def init_database(database_url: str = None, echo: bool = False) -> DatabaseConnection:
    """
    Initialize the database connection.
    
    Call this in your app startup.
    
    Args:
        database_url: PostgreSQL connection URL
        echo: If True, log SQL statements
        
    Returns:
        DatabaseConnection instance
    """
    global _db_instance
    _db_instance = DatabaseConnection(database_url=database_url, echo=echo)
    return _db_instance
