"""
Test script for streaming output in CopilotAPI, especially with MCP tool calls.

This script tests the streaming output functionality added to the CopilotAPI.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, AsyncGenerator, Tuple

from app.client.enhanced_copilot import EnhancedCopilotClient

# Sample messages with MCP tool queries
test_messages = [
    # Regular query
    [
        {"role": "user", "content": "Tell me about Claude AI"}
    ],
    
    # MCP tool query (web search)
    [
        {"role": "user", "content": "Search for the latest AI news"},
        {"role": "assistant", "content": "I'll search for the latest AI news for you."},
        {"role": "user", "content": "Use the MCP tool to search"}
    ],
    
    # Malformed MCP tool query
    [
        {"role": "user", "content": "Search for AI news today"},
        {"role": "assistant", "content": "<use_mcp_tool_name>@smithery-ai</server_name>\n<tool_name>brave_web_search</tool_name>\n{\n  \"query\": \"artificial intelligence AI news today\",\n  \"count\": 5>\n</use_mcp_tool>"}
    ]
]

async def test_streaming():
    """Test the streaming output functionality."""
    client = EnhancedCopilotClient()
    
    print("Testing streaming output...\n")
    
    for i, messages in enumerate(test_messages):
        print(f"\nTest {i+1}: {messages[-1]['content'][:50]}...\n")
        print("-" * 80)
        
        # Collect stream output
        content_parts = []
        mcp_tool_parts = []
        
        try:
            async for event_type, event_data in client.generate_chat_completion(
                messages=messages,
                stream=True
            ):
                if event_type == "content":
                    content = event_data.get("content", "")
                    content_parts.append(content)
                    print(f"CONTENT: {content}")
                elif event_type == "mcp_tool":
                    mcp_content = event_data.get("content", "")
                    mcp_tool_parts.append(mcp_content)
                    print(f"MCP TOOL: {mcp_content[:100]}...")
                elif event_type == "error":
                    print(f"ERROR: {event_data.get('error', '')}")
                elif event_type == "done":
                    print("DONE")
        except Exception as e:
            print(f"Exception during streaming: {e}")
        
        # Print summary
        print("\nSummary:")
        print(f"- Content chunks: {len(content_parts)}")
        print(f"- MCP tool responses: {len(mcp_tool_parts)}")
        print("-" * 80)
        
        # Brief pause between tests
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(test_streaming())
