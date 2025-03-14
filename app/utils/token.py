"""Token management utility for GitHub Copilot API."""

import json
import time
import asyncio
from typing import Dict, Any, Optional

import httpx

from app.utils.logger import logger
from app.config.settings import get_settings

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
    
    # If we already have a config, check if it's still valid
    if settings.copilot_config:
        expires_at = settings.copilot_config.get("expires_at", 0)
        current_time = int(time.time())
        
        # If token is valid for more than 2 minutes, use it
        if expires_at > current_time + 120:
            return settings.copilot_config
    
    # Need to fetch a new token (asynchronously)
    return await fetch_copilot_token()

def get_token_config() -> Optional[Dict[str, Any]]:
    """
    Synchronous wrapper for getting token config. Use only in sync code.
    
    Returns:
        Optional[Dict[str, Any]]: Token configuration or None if failed
    """
    settings = get_settings()
    
    # If we already have a config, check if it's still valid
    if settings.copilot_config:
        expires_at = settings.copilot_config.get("expires_at", 0)
        current_time = int(time.time())
        
        # If token is valid for more than 2 minutes, use it
        if expires_at > current_time + 120:
            return settings.copilot_config
    
    # For synchronous contexts, we need a more careful approach
    # This function should only be called at startup or in sync code
    logger.warning("Synchronous token fetch may block the application")
    
    # We'll return None and let the caller handle it
    # The token will be fetched asynchronously on first request
    return None

def is_token_expiring_soon(config: Dict[str, Any], threshold_seconds: int = 120) -> bool:
    """
    Check if the token is expiring soon.
    
    Args:
        config: Token configuration
        threshold_seconds: Threshold in seconds
        
    Returns:
        bool: True if token expires within threshold_seconds
    """
    expires_at = config.get("expires_at", 0)
    current_time = int(time.time())
    
    return expires_at - current_time < threshold_seconds
