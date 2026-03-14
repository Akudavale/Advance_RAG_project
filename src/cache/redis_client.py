"""
src/cache/redis_client.py
-------------------------
Redis client wrapper for conversation caching.

Features:
- Connection pooling
- Key prefix management
- Automatic TTL handling
- Serialization utilities
"""

import logging
import json
import os
from typing import Any, Optional, List, Dict

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Wrapper for Redis operations.
    
    Handles:
    - Connection management
    - Key prefixing
    - Serialization/deserialization
    - Error handling
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "rag:",
        default_ttl: int = 3600
    ):
        """
        Initialize Redis client.
        
        Args:
            host: Redis host (default: localhost, from REDIS_HOST env)
            port: Redis port (default: 6379, from REDIS_PORT env)
            db: Redis database number
            password: Redis password (from REDIS_PASSWORD env)
            prefix: Key prefix for all keys
            default_ttl: Default time-to-live in seconds (1 hour)
        """
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis package not found. Install with: pip install redis"
            )
        
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = int(port or os.getenv("REDIS_PORT", 6379))
        self.db = db
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.prefix = prefix
        self.default_ttl = default_ttl
        
        logger.info(
            f"Connecting to Redis: {self.host}:{self.port}/db{self.db}"
        )
        
        try:
            # Create connection pool
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("✓ Redis connection successful")
            
        except Exception as e:
            logger.error(f"✗ Redis connection failed: {e}")
            raise
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        json_serialize: bool = True
    ) -> bool:
        """
        Set a key-value pair.
        
        Args:
            key: Key name
            value: Value (will be JSON serialized if json_serialize=True)
            ttl: Time-to-live in seconds (uses default if None)
            json_serialize: If True, JSON serialize the value
            
        Returns:
            True if successful
        """
        try:
            full_key = self._make_key(key)
            ttl = ttl or self.default_ttl
            
            if json_serialize:
                value = json.dumps(value)
            
            self.redis_client.setex(full_key, ttl, value)
            logger.debug(f"Redis SET: {key} (ttl={ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Redis SET failed: {e}")
            return False
    
    def get(
        self,
        key: str,
        json_deserialize: bool = True
    ) -> Optional[Any]:
        """
        Get a value by key.
        
        Args:
            key: Key name
            json_deserialize: If True, JSON deserialize the value
            
        Returns:
            Value or None if not found
        """
        try:
            full_key = self._make_key(key)
            value = self.redis_client.get(full_key)
            
            if value is None:
                logger.debug(f"Redis MISS: {key}")
                return None
            
            logger.debug(f"Redis HIT: {key}")
            
            if json_deserialize and isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            
            return value
            
        except Exception as e:
            logger.error(f"Redis GET failed: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key."""
        try:
            full_key = self._make_key(key)
            self.redis_client.delete(full_key)
            logger.debug(f"Redis DELETE: {key}")
            return True
        except Exception as e:
            logger.error(f"Redis DELETE failed: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            full_key = self._make_key(key)
            return self.redis_client.exists(full_key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS failed: {e}")
            return False
    
    def incr(self, key: str, amount: int = 1) -> int:
        """Increment a counter."""
        try:
            full_key = self._make_key(key)
            return self.redis_client.incrby(full_key, amount)
        except Exception as e:
            logger.error(f"Redis INCR failed: {e}")
            return 0
    
    def lpush(self, key: str, *values) -> int:
        """Push values to list (left side)."""
        try:
            full_key = self._make_key(key)
            for value in values:
                if not isinstance(value, str):
                    value = json.dumps(value)
                self.redis_client.lpush(full_key, value)
            self.redis_client.expire(full_key, self.default_ttl)
            logger.debug(f"Redis LPUSH: {key}")
            return len(values)
        except Exception as e:
            logger.error(f"Redis LPUSH failed: {e}")
            return 0
    
    def lrange(
        self,
        key: str,
        start: int = 0,
        stop: int = -1,
        json_deserialize: bool = True
    ) -> List[Any]:
        """Get range from list."""
        try:
            full_key = self._make_key(key)
            values = self.redis_client.lrange(full_key, start, stop)
            
            if json_deserialize:
                deserialized = []
                for value in values:
                    try:
                        deserialized.append(json.loads(value))
                    except json.JSONDecodeError:
                        deserialized.append(value)
                values = deserialized
            
            logger.debug(f"Redis LRANGE: {key} ({len(values)} items)")
            return values
            
        except Exception as e:
            logger.error(f"Redis LRANGE failed: {e}")
            return []
    
    def ltrim(self, key: str, start: int = 0, stop: int = -1) -> bool:
        """Trim list to range."""
        try:
            full_key = self._make_key(key)
            self.redis_client.ltrim(full_key, start, stop)
            logger.debug(f"Redis LTRIM: {key}")
            return True
        except Exception as e:
            logger.error(f"Redis LTRIM failed: {e}")
            return False
    
    def hset(
        self,
        key: str,
        mapping: Dict[str, Any],
        json_serialize: bool = True
    ) -> int:
        """Set hash fields."""
        try:
            full_key = self._make_key(key)
            
            if json_serialize:
                mapping = {
                    k: json.dumps(v) if not isinstance(v, str) else v
                    for k, v in mapping.items()
                }
            
            result = self.redis_client.hset(full_key, mapping=mapping)
            self.redis_client.expire(full_key, self.default_ttl)
            logger.debug(f"Redis HSET: {key}")
            return result
            
        except Exception as e:
            logger.error(f"Redis HSET failed: {e}")
            return 0
    
    def hgetall(
        self,
        key: str,
        json_deserialize: bool = True
    ) -> Dict[str, Any]:
        """Get all hash fields."""
        try:
            full_key = self._make_key(key)
            mapping = self.redis_client.hgetall(full_key)
            
            if json_deserialize:
                deserialized = {}
                for k, v in mapping.items():
                    try:
                        deserialized[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        deserialized[k] = v
                mapping = deserialized
            
            logger.debug(f"Redis HGETALL: {key}")
            return mapping
            
        except Exception as e:
            logger.error(f"Redis HGETALL failed: {e}")
            return {}
    
    def close(self):
        """Close Redis connection."""
        try:
            self.redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis: {e}")


# Global Redis instance
_redis_instance = None


def get_redis_client(
    host: str = None,
    port: int = None,
    **kwargs
) -> Optional[RedisClient]:
    """
    Get or create the global Redis client.
    
    Args:
        host: Redis host
        port: Redis port
        **kwargs: Additional arguments for RedisClient
        
    Returns:
        RedisClient instance or None if connection fails
    """
    global _redis_instance
    
    if _redis_instance is None:
        try:
            _redis_instance = RedisClient(host=host, port=port, **kwargs)
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            return None
    
    return _redis_instance


def init_redis(
    host: str = None,
    port: int = None,
    **kwargs
) -> Optional[RedisClient]:
    """
    Initialize Redis connection.
    
    Call this at app startup.
    
    Args:
        host: Redis host
        port: Redis port
        **kwargs: Additional arguments
        
    Returns:
        RedisClient instance or None if connection fails
    """
    global _redis_instance
    
    try:
        _redis_instance = RedisClient(host=host, port=port, **kwargs)
        return _redis_instance
    except Exception as e:
        logger.warning(f"Redis initialization failed: {e}")
        return None
