"""Token management utility for GitHub Copilot API with Redis storage."""

import json
import time
import asyncio
from typing import Dict, Any, Optional

import httpx

from app.utils.logger import logger
from app.utils.redis_manager import RedisManager
from app.config.settings import get_settings

# Initialize Redis manager
redis_manager = RedisManager()

async def fetch_copilot_token() -> Optional[Dict[str, Any]]:
    """
    Fetch a new GitHub Copilot token configuration from the API.
    
    Returns:
        Optional[Dict[str, Any]]: Token configuration or None if failed
    """
    settings = get_settings()
    github_token = settings.github_copilot_token
    
    if not github_token:
        logger.error("Missing GitHub Copilot token in environment variables")
        return None
    
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/json",
        "Copilot-Integration-Id": "vscode-chat",
        "Editor-Version": "Python/1.0.0"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.github.com/copilot_internal/v2/token",
                headers=headers,
                timeout=10.0
            )
            
            response.raise_for_status()
            config = response.json()
            
            # Store the config in Redis
            if redis_manager.is_connected():
                redis_manager.set_copilot_config(config)
                logger.info("Stored new Copilot token in Redis")
            
            # Update the settings
            settings.copilot_config = config
            
            logger.info("Successfully fetched GitHub Copilot token configuration")
            return config
            
    except httpx.RequestError as e:
        logger.error(f"Error fetching GitHub Copilot token: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching GitHub Copilot token: {e}")
        return None

async def get_token_config_async() -> Optional[Dict[str, Any]]:
    """
    Asynchronously get the current token configuration, fetching a new one if needed.
    
    Returns:
        Optional[Dict[str, Any]]: Token configuration or None if failed
    """
    settings = get_settings()
    current_time = int(time.time())
    
    # Try to get config from settings first
    config = settings.copilot_config
    
    # If no config in settings, try to get from Redis
    if not config and redis_manager.is_connected():
        config = redis_manager.get_copilot_config()
        if config:
            # Update settings with config from Redis
            settings.copilot_config = config
            logger.debug("Retrieved Copilot configuration from Redis")
    
    # If we have a config, check if it's still valid
    if config:
        expires_at = config.get("expires_at", 0)
        
        # Check if the token is about to expire (3 minutes buffer)
        if expires_at > current_time + 180:
            # Token is still valid
            return config
        else:
            logger.info(f"Token expires in {expires_at - current_time} seconds, refreshing")
    
    # Need to fetch a new token
    return await fetch_copilot_token()

def get_token_config() -> Optional[Dict[str, Any]]:
    """
    Synchronous wrapper for getting token config. Use only in sync code.
    
    Returns:
        Optional[Dict[str, Any]]: Token configuration or None if failed
    """
    settings = get_settings()
    current_time = int(time.time())
    
    # Try to get config from settings first
    config = settings.copilot_config
    
    # If no config in settings, try to get from Redis
    if not config and redis_manager.is_connected():
        config = redis_manager.get_copilot_config()
        if config:
            # Update settings with config from Redis
            settings.copilot_config = config
            logger.debug("Retrieved Copilot configuration from Redis")
    
    # If we have a config, check if it's still valid
    if config:
        expires_at = config.get("expires_at", 0)
        
        # Check if the token is about to expire (3 minutes buffer)
        if expires_at > current_time + 180:
            # Token is still valid
            return config
    
    # For synchronous contexts, we need a more careful approach
    # This function should only be called at startup or in sync code
    logger.warning("Synchronous token fetch may block the application")
    
    # We'll return None and let the caller handle it
    # The token will be fetched asynchronously on first request
    return None

def is_token_expiring_soon(config: Dict[str, Any], threshold_seconds: int = 180) -> bool:
    """
    Check if the token is expiring soon.
    
    Args:
        config: Token configuration
        threshold_seconds: Threshold in seconds (default: 3 minutes)
        
    Returns:
        bool: True if token expires within threshold_seconds
    """
    expires_at = config.get("expires_at", 0)
    current_time = int(time.time())
    
    return expires_at - current_time < threshold_seconds

async def schedule_token_refresh():
    """
    Background task to periodically check token validity and refresh if needed.
    This should be started at application startup.
    """
    while True:
        try:
            # Get the current token configuration
            config = await get_token_config_async()
            
            if config:
                # Calculate time until token expires
                expires_at = config.get("expires_at", 0)
                current_time = int(time.time())
                time_until_expiry = expires_at - current_time
                
                # If token expires in less than 5 minutes but more than 3 minutes,
                # sleep until the 3-minute threshold and then refresh
                if 180 < time_until_expiry < 300:
                    wait_time = time_until_expiry - 180
                    logger.debug(f"Token expires in {time_until_expiry} seconds, waiting {wait_time} seconds before refreshing")
                    await asyncio.sleep(wait_time)
                    await fetch_copilot_token()
                elif time_until_expiry <= 180:
                    # Token expires in less than 3 minutes, refresh immediately
                    logger.info("Token expires soon, refreshing immediately")
                    await fetch_copilot_token()
                else:
                    # Token is still valid, sleep for a while before checking again
                    # We'll sleep for half the time until we hit the 5-minute threshold,
                    # or maximum 1 hour
                    sleep_time = min((time_until_expiry - 300) / 2, 3600)
                    logger.debug(f"Token valid for {time_until_expiry} seconds, sleeping for {sleep_time} seconds")
                    await asyncio.sleep(sleep_time)
            else:
                # No valid token, try to fetch one and then sleep for a while
                await fetch_copilot_token()
                await asyncio.sleep(60)  # Sleep for 1 minute before checking again
                
        except Exception as e:
            logger.error(f"Error in token refresh background task: {e}")
            await asyncio.sleep(60)  # Sleep for 1 minute before retrying
