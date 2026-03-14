#!/usr/bin/env python
"""
scripts/init_databases.py
------------------------
Initialize PostgreSQL and Redis connections for the RAG project.

Usage:
    python scripts/init_databases.py
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from src.database import init_database, get_database
from src.cache import init_redis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_databases():
    """Initialize PostgreSQL and Redis."""
    
    config = Config()
    logger.info("Initializing databases...")
    logger.info(f"Environment: DB_URL={config._mask_url(config.DB_URL) if hasattr(config, '_mask_url') else '***'}")
    
    # Initialize PostgreSQL
    if config.ENABLE_POSTGRES:
        logger.info("\n[1/2] Initializing PostgreSQL...")
        try:
            db = init_database(config.DB_URL, echo=config.DB_ECHO)
            logger.info("✓ PostgreSQL initialized successfully")
            logger.info(f"  Connection: {config._mask_url(config.DB_URL)}")
            logger.info("  Tables: conversations, messages, documents, user_feedback, etc.")
        except Exception as e:
            logger.error(f"✗ PostgreSQL initialization failed: {e}")
            return False
    else:
        logger.warning("PostgreSQL disabled (ENABLE_POSTGRES=false)")
    
    # Initialize Redis
    if config.ENABLE_REDIS:
        logger.info("\n[2/2] Initializing Redis...")
        try:
            redis = init_redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                prefix="rag:",
                default_ttl=config.REDIS_DEFAULT_TTL
            )
            if redis:
                logger.info("✓ Redis initialized successfully")
                logger.info(f"  Connection: {config.REDIS_HOST}:{config.REDIS_PORT}")
                logger.info(f"  Default TTL: {config.REDIS_DEFAULT_TTL}s")
            else:
                logger.warning("Redis not available (optional)")
        except Exception as e:
            logger.warning(f"Redis initialization failed (optional): {e}")
    else:
        logger.warning("Redis disabled (ENABLE_REDIS=false)")
    
    logger.info("\n✓ Database initialization complete!")
    return True


if __name__ == "__main__":
    success = init_databases()
    sys.exit(0 if success else 1)
