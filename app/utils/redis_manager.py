"""Redis manager for storing and retrieving GitHub Copilot tokens."""

import os
import json
import time
from typing import Dict, Any, Optional

import redis
from app.utils.logger import logger

# Redis key constants
REDIS_KEY_COPILOT_CONFIG = "copilot_api:token_config"


class RedisManager:
    """Redis manager for token storage and retrieval with singleton pattern."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(RedisManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Redis connection using environment variables."""
        if self._initialized:
            return

        # Get Redis configuration from environment
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))
        redis_password = os.getenv("REDIS_PASSWORD", None)

        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=True,  # Automatically decode responses to strings
                socket_timeout=5,  # 5 second timeout
                socket_connect_timeout=5,
            )
            self._initialized = True
            logger.debug(f"Redis connection established: {redis_host}:{redis_port}/{redis_db}")
        except Exception as e:
            logger.error(f"Redis connection failed: {str(e)}")
            # If Redis connection fails, we'll continue using environment variables
            self.redis_client = None

    def is_connected(self) -> bool:
        """Check if Redis connection is active."""
        if not self.redis_client:
            return False
        try:
            return self.redis_client.ping()
        except Exception:
            return False

    def get_copilot_config(self) -> Optional[Dict[str, Any]]:
        """
        Get Copilot configuration from Redis.
        
        Returns:
            Optional[Dict[str, Any]]: Copilot configuration or None if not found
        """
        if not self.is_connected():
            logger.warning("Redis unavailable, cannot get Copilot configuration")
            return None

        try:
            config_json = self.redis_client.get(REDIS_KEY_COPILOT_CONFIG)
            if not config_json:
                logger.debug("No Copilot configuration found in Redis")
                return None

            config = json.loads(config_json)
            logger.debug(f"Retrieved Copilot configuration from Redis, expires at: {config.get('expires_at', 0)}")
            return config
        except Exception as e:
            logger.error(f"Error retrieving Copilot configuration from Redis: {str(e)}")
            return None

    def set_copilot_config(self, config: Dict[str, Any]) -> bool:
        """
        Store Copilot configuration in Redis.
        
        Args:
            config: Copilot configuration to store
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.warning("Redis unavailable, cannot store Copilot configuration")
            return False

        try:
            config_json = json.dumps(config)
            self.redis_client.set(REDIS_KEY_COPILOT_CONFIG, config_json)
            
            # Set expiration time for the key (token expiry + 1 hour)
            if 'expires_at' in config:
                # Calculate TTL (expiry time - current time + 1 hour buffer)
                current_time = int(time.time())
                ttl = max(0, config['expires_at'] - current_time) + 3600
                self.redis_client.expire(REDIS_KEY_COPILOT_CONFIG, ttl)
                
            logger.debug("Copilot configuration stored in Redis")
            return True
        except Exception as e:
            logger.error(f"Error storing Copilot configuration in Redis: {str(e)}")
            return False
