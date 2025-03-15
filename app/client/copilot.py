"""Client for GitHub Copilot API."""

import json
import time
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple

import aiohttp
from aiohttp.client_exceptions import ClientError

from app.utils.logger import logger
from app.utils.token import get_token_config_async, is_token_expiring_soon, fetch_copilot_token


class CopilotClient:
    """Client for interacting with GitHub Copilot API."""
    
    def __init__(
        self, 
        model: Optional[str] = None,
        fallback_model: Optional[str] = None
    ):
        """
        Initialize the CopilotClient.
        
        Args:
            model: Model name to use (default to claude-3.7-sonnet if None)
            fallback_model: Fallback model name to use on rate limit errors
        """
        self.model = model or "claude-3.7-sonnet"
        self.fallback_model = fallback_model
        self.api_url = "https://api.githubcopilot.com/chat/completions"
        self._config = None
    
    async def ensure_valid_token(self) -> bool:
        """
        Ensure we have a valid token configuration.
        
        Returns:
            bool: True if valid token is available
        """
        # Get current config or fetch new one asynchronously
        if self._config is None:
            self._config = await get_token_config_async()
        
        if not self._config:
            # Try fetching asynchronously
            self._config = await fetch_copilot_token()
            if not self._config:
                logger.error("Failed to fetch token configuration")
                return False
        
        # Check if token is about to expire
        if is_token_expiring_soon(self._config):
            logger.info("Token expiring soon, fetching new token")
            self._config = await fetch_copilot_token()
            if not self._config:
                logger.error("Failed to refresh expiring token")
                return False
        
        return True
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for the API request.
        
        Returns:
            Dict[str, str]: HTTP headers
        """
        if not self._config:
            raise ValueError("No valid token configuration available")
        
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config['token']}",
            "Copilot-Integration-Id": "vscode-chat",
            "Editor-Version": "vscode/1.85.1",
            "Editor-Plugin-Version": "copilot/1.171.0",
            "Accept": "application/json",
            "X-Tracking-Id": self._config.get("tracking_id", ""),
            "X-Copilot-Repository": "copilot-api"
        }
    
    async def _make_request(self, headers: Dict[str, str], data: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
        """
        Make a request to the Copilot API.
        
        Args:
            headers: HTTP headers
            data: Request data
            
        Yields:
            bytes: Response chunks
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=data) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise ClientError(f"HTTP error {response.status}: {error_text}")
                
                # Use larger chunk size to reduce the chance of splitting JSON or XML structures
                async for chunk in response.content.iter_chunked(4096):
                    if chunk:
                        yield chunk
    
    async def _make_request_with_fallback(
        self, 
        headers: Dict[str, str], 
        data: Dict[str, Any],
        original_model: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Make a request with fallback to another model on rate limit errors.
        
        Args:
            headers: HTTP headers
            data: Request data
            original_model: Original model name
            
        Yields:
            bytes: Response chunks
        """
        try:
            # Try with original model
            async for chunk in self._make_request(headers, data):
                yield chunk
        except ClientError as e:
            # Check if it's a rate limit error
            if "429" in str(e) and self.fallback_model:
                logger.warning(f"Rate limit error, falling back to {self.fallback_model}")
                
                # Switch to fallback model
                data["model"] = self.fallback_model
                
                # Adjust parameters for Claude if needed
                if "top_p" in data and data["top_p"] > 1.0:
                    data["top_p"] = 1.0
                if "temperature" in data and data["temperature"] > 1.0:
                    data["temperature"] = 1.0
                
                try:
                    # Retry with fallback model
                    async for chunk in self._make_request(headers, data):
                        yield chunk
                    logger.info(f"Successfully completed request with {self.fallback_model}")
                    return
                except Exception as retry_error:
                    logger.error(f"Fallback to {self.fallback_model} also failed: {retry_error}")
                    raise e
            else:
                # Not a rate limit error or no fallback model
                raise
    
    async def generate_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stream: bool = True
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """
        Generate a chat completion from Copilot.
        
        Args:
            messages: List of message objects with role and content
            model: Model name (overrides instance model)
            temperature: Temperature for generation
            top_p: Top-p for nucleus sampling
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            stream: Whether to stream the response
            
        Yields:
            Tuple[str, Dict[str, Any]]: Event type and content
        """
        # Ensure we have a valid token
        valid = await self.ensure_valid_token()
        if not valid:
            yield "error", {"error": "Failed to obtain valid token"}
            return
        
        # Get headers
        headers = self._get_headers()
        
        # Use specified model or instance model
        use_model = model or self.model
        logger.debug(f"Using model: {use_model}")
        
        # Prepare request data
        data = {
            "model": use_model,
            "messages": messages,
            "temperature": float(temperature),
            "stream": True,  # Always use streaming for consistent handling
            "response_format": {"type": "text"},
            "max_tokens": 8192,
            "top_p": float(top_p),
            "presence_penalty": float(presence_penalty),
            "frequency_penalty": float(frequency_penalty)
        }
        
        try:
            # Make request with fallback
            async for chunk in self._make_request_with_fallback(headers, data, use_model):
                chunk_str = chunk.decode("utf-8", errors="replace")
                if not chunk_str.strip():
                    continue
                
                for line in chunk_str.split("\n"):
                    if line.startswith("data: "):
                        json_str = line[6:]  # Remove 'data: ' prefix
                        if json_str.strip() == "[DONE]":
                            yield "done", {}
                            return
                        
                        try:
                            data = json.loads(json_str)
                            # Extract content from the response
                            content = (
                                data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            
                            if content:
                                yield "content", {"content": content}
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            yield "error", {"error": str(e)}
    
    async def complete_chat_without_streaming(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0
    ) -> Dict[str, Any]:
        """
        Complete a chat without streaming (collects the full response).
        
        Args:
            messages: List of message objects
            model: Model to use
            temperature: Temperature
            top_p: Top-p
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            
        Returns:
            Dict[str, Any]: Complete response
        """
        complete_response = []
        
        async for event_type, event_data in self.generate_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stream=False
        ):
            if event_type == "content":
                complete_response.append(event_data.get("content", ""))
            elif event_type == "error":
                raise ValueError(event_data.get("error", "Unknown error"))
        
        return {"content": "".join(complete_response)}
