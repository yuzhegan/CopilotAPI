"""Enhanced client for GitHub Copilot API with improved streaming and MCP support."""

import json
import time
import re
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple

import aiohttp
from aiohttp.client_exceptions import ClientError

from app.utils.logger import logger
from app.utils.token import get_token_config_async, is_token_expiring_soon, fetch_copilot_token


class EnhancedCopilotClient:
    """Enhanced client for interacting with GitHub Copilot API."""
    
    def __init__(
        self, 
        model: Optional[str] = None,
        fallback_model: Optional[str] = None
    ):
        """
        Initialize the EnhancedCopilotClient.
        
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
    
    def _process_messages_for_mcp(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process messages to ensure proper XML formatting for MCP tool calls.
        This fixes issues with malformed XML in tool calls.
        
        Args:
            messages: Original messages
            
        Returns:
            List[Dict[str, Any]]: Processed messages with fixed XML formatting
        """
        processed_messages = []
        
        for message in messages:
            content = message.get("content", "")
            
            # Check if this is potentially an MCP tool call
            if "<use_mcp_tool" in content:
                # Fix common XML formatting issues
                # 1. Fix incomplete or malformed tags
                content = self._fix_mcp_tool_xml(content)
                
                # Update the message with fixed content
                processed_message = message.copy()
                processed_message["content"] = content
                processed_messages.append(processed_message)
            else:
                # No changes needed
                processed_messages.append(message)
        
        return processed_messages
    
    def _fix_mcp_tool_xml(self, content: str) -> str:
        """
        Fix common XML formatting issues in MCP tool calls.
        
        Args:
            content: Original content with potentially malformed XML
            
        Returns:
            str: Fixed content with proper XML formatting
        """
        # Lines in the content
        lines = content.split("\n")
        xml_sections = []
        current_section = []
        in_tool_section = False
        
        # Find and isolate MCP tool sections
        for line in lines:
            if "<use_mcp_tool" in line:
                in_tool_section = True
                current_section = [line]
            elif in_tool_section:
                current_section.append(line)
                if "</use_mcp_tool>" in line:
                    in_tool_section = False
                    xml_sections.append(current_section)
                    current_section = []
            else:
                xml_sections.append([line])
        
        # If we're still in a tool section at the end, add it
        if in_tool_section and current_section:
            xml_sections.append(current_section)
        
        # Process each XML section
        processed_sections = []
        for section in xml_sections:
            if any("<use_mcp_tool" in line for line in section):
                processed_section = self._process_mcp_tool_section(section)
                processed_sections.append(processed_section)
            else:
                processed_sections.append("\n".join(section))
        
        return "\n".join(processed_sections)
    
    def _process_mcp_tool_section(self, section: List[str]) -> str:
        """
        Process an MCP tool section to fix XML formatting issues.
        
        Args:
            section: Lines of the MCP tool section
            
        Returns:
            str: Processed section with fixed XML formatting
        """
        # Join the section lines
        section_text = "\n".join(section)
        
        # Standard format for MCP tool call
        standard_format = """<use_mcp_tool>
<server_name>{server_name}</server_name>
<tool_name>{tool_name}</tool_name>
<arguments>
{arguments}
</arguments>
</use_mcp_tool>"""
        
        # Extract components from the malformed XML
        server_name = self._extract_between_tags(section_text, "server_name")
        if not server_name:
            # Try to find server name with other tag patterns
            for pattern in ["<server_name>", "server_name=", "@smithery-ai"]:
                if pattern in section_text:
                    server_name = "@smithery-ai-brave-search"
                    break
        
        tool_name = self._extract_between_tags(section_text, "tool_name")
        if not tool_name and "brave_web_search" in section_text:
            tool_name = "brave_web_search"
        
        # Extract arguments - typically JSON
        arguments = ""
        if "{" in section_text and "}" in section_text:
            start = section_text.find("{")
            end = section_text.rfind("}") + 1
            if start < end:
                try:
                    # Parse and re-format to ensure valid JSON
                    args_text = section_text[start:end]
                    args_dict = json.loads(args_text)
                    arguments = json.dumps(args_dict, indent=2)
                except json.JSONDecodeError:
                    # Fallback: try to construct a valid JSON
                    arguments = self._construct_fallback_json(section_text)
        
        # If we couldn't extract all necessary components, return original
        if not server_name or not tool_name or not arguments:
            logger.warning("Could not extract all MCP tool components, using original")
            return section_text
        
        # Format with the extracted components
        return standard_format.format(
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments
        )
    
    def _extract_between_tags(self, text: str, tag_name: str) -> str:
        """
        Extract content between opening and closing tags.
        
        Args:
            text: Text to search in
            tag_name: Name of the tag
            
        Returns:
            str: Content between tags, or empty string if not found
        """
        start_tag = f"<{tag_name}>"
        end_tag = f"</{tag_name}>"
        
        start = text.find(start_tag)
        if start == -1:
            return ""
        
        start += len(start_tag)
        end = text.find(end_tag, start)
        
        if end == -1:
            return ""
        
        return text[start:end].strip()
    
    def _construct_fallback_json(self, text: str) -> str:
        """
        Attempt to construct a valid JSON for arguments when parsing fails.
        
        Args:
            text: Original text with malformed JSON
            
        Returns:
            str: Constructed JSON string
        """
        # Look for query and count patterns
        query = ""
        count = 10
        
        # Try to extract query
        query_patterns = [r'query":\s*"([^"]+)"', r"query':\s*'([^']+)'", r'query:\s*([^,\n}]+)'] 
        for pattern in query_patterns:
            match = re.search(pattern, text)
            if match:
                query = match.group(1).strip()
                break
        
        # Try to extract count
        count_patterns = [r'count":\s*(\d+)', r"count':\s*(\d+)", r'count:\s*(\d+)'] 
        for pattern in count_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    count = int(match.group(1))
                    break
                except ValueError:
                    pass
        
        # Construct JSON
        result = {
            "query": query or "artificial intelligence AI news today",
            "count": count
        }
        
        return json.dumps(result, indent=2)
    
    def _process_mcp_tool_response(self, content: str) -> str:
        """
        Process MCP tool response content into a more structured format.
        
        Args:
            content: Raw MCP tool response
            
        Returns:
            str: Processed response
        """
        # Look for common MCP tool response patterns
        if "<use_mcp_tool>" in content and "</use_mcp_tool>" in content:
            # This is likely a properly formatted MCP tool call
            # Just return it as is for now
            return content
        
        # If we have fragments, try to piece them together
        if "<use_mcp_tool" in content and "</use_mcp_tool>" not in content:
            # This is an incomplete MCP tool call
            # Try to identify the server and tool
            server_name = ""
            tool_name = ""
            
            if "<server_name>" in content:
                server_name = self._extract_between_tags(content, "server_name")
            elif "server_name=" in content or "@smithery-ai" in content:
                server_name = "@smithery-ai-brave-search"
            
            if "<tool_name>" in content:
                tool_name = self._extract_between_tags(content, "tool_name")
            elif "brave_web_search" in content:
                tool_name = "brave_web_search"
            
            # Extract arguments
            arguments = ""
            if "{" in content and "}" in content:
                try:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    args_text = content[start:end]
                    args_dict = json.loads(args_text)
                    arguments = json.dumps(args_dict, indent=2)
                except (json.JSONDecodeError, ValueError):
                    arguments = self._construct_fallback_json(content)
            
            # If we have enough information, construct a proper MCP tool call
            if server_name and tool_name and arguments:
                return f"""<use_mcp_tool>
<server_name>{server_name}</server_name>
<tool_name>{tool_name}</tool_name>
<arguments>
{arguments}
</arguments>
</use_mcp_tool>"""
        
        # If all else fails, return the original content
        return content
    
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
        Generate a chat completion from Copilot with enhanced streaming support.
        
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
        
        # Process messages for MCP tool formatting issues
        processed_messages = self._process_messages_for_mcp(messages)

        # Prepare request data
        data = {
            "model": use_model,
            "messages": processed_messages,
            "temperature": float(temperature),
            "stream": True,  # Always use streaming for consistent handling
            "response_format": {"type": "text"},
            "max_tokens": 8192,
            "top_p": float(top_p),
            "presence_penalty": float(presence_penalty),
            "frequency_penalty": float(frequency_penalty)
        }
        
        # Initialize stream processing variables
        buffer = ""
        in_mcp_tool = False
        mcp_tool_buffer = ""
        
        try:
            # Make request with fallback
            async for chunk in self._make_request_with_fallback(headers, data, use_model):
                chunk_str = chunk.decode("utf-8", errors="replace")
                if not chunk_str.strip():
                    continue
                
                # Add to buffer
                buffer += chunk_str
                
                # Process complete data lines
                while True:
                    # Find start of a data line
                    data_start = buffer.find("data: ")
                    if data_start == -1:
                        break
                    
                    # Find end of this data line
                    line_end = buffer.find("\n", data_start)
                    if line_end == -1:
                        # Incomplete line, wait for more data
                        break
                    
                    # Extract this data line
                    line = buffer[data_start:line_end].strip()
                    # Remove processed part from buffer
                    buffer = buffer[line_end + 1:]
                    
                    # Handle MCP tool response detection and accumulation
                    if "<use_mcp_tool" in line or in_mcp_tool:
                        if not in_mcp_tool:
                            logger.debug("Detected start of MCP tool response")
                            in_mcp_tool = True
                            mcp_tool_buffer = ""
                        
                        # Add to MCP tool buffer (removing data: prefix)
                        if line.startswith("data: "):
                            mcp_tool_buffer += line[6:] + "\n"
                        else:
                            mcp_tool_buffer += line + "\n"
                        
                        # Check if MCP tool response is complete
                        if "</use_mcp_tool>" in line:
                            in_mcp_tool = False
                            logger.debug("MCP tool response complete")
                            
                            # Process the MCP tool response
                            processed_response = self._process_mcp_tool_response(mcp_tool_buffer)
                            yield "mcp_tool", {"content": processed_response}
                        
                        # Continue to next line without further processing
                        continue
                    
                    # Handle regular data lines
                    if line.startswith("data: "):
                        json_str = line[6:]  # Remove 'data: ' prefix
                        
                        # Check for end of stream
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
                        except json.JSONDecodeError as json_error:
                            logger.warning(f"JSON decode error: {json_error} - Line: {json_str[:100]}...")
                            
                            # Check if it might be the start of an MCP tool response
                            if "<use_mcp_tool" in json_str:
                                in_mcp_tool = True
                                mcp_tool_buffer = json_str + "\n"
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
        mcp_tool_responses = []
        
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
            elif event_type == "mcp_tool":
                mcp_tool_responses.append(event_data.get("content", ""))
            elif event_type == "error":
                raise ValueError(event_data.get("error", "Unknown error"))
        
        # Combine all responses
        full_content = "".join(complete_response)
        
        # If there were MCP tool responses, add them
        if mcp_tool_responses:
            if full_content:
                full_content += "\n\n"
            full_content += "\n\n".join(mcp_tool_responses)
        
        return {"content": full_content}
