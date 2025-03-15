"""
Test script for enhanced streaming with MCP tool support in CopilotAPI.

This script allows you to test the improved streaming output, particularly
with MCP tool calls, in an interactive way.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, AsyncGenerator, Tuple

from app.client.enhanced_copilot import EnhancedCopilotClient

# Sample messages to use in tests
TEST_MESSAGES = [
    # Simple content test
    [
        {"role": "user", "content": "Give me a brief introduction to artificial intelligence"}
    ],
    
    # Web search test with proper MCP tool format
    [
        {"role": "user", "content": "Search for the latest AI research news"},
        {"role": "assistant", "content": "I'll use MCP to search for the latest AI research news."},
        {"role": "user", "content": "Please go ahead and use MCP to search."}
    ],
    
    # Test with malformed MCP tool call
    [
        {"role": "user", "content": "Search for recent AI breakthroughs"},
        {"role": "assistant", "content": "<use_mcp_tool_name>@smithery-ai</server_name>\n<tool_name>brave_web_search</tool_name>\n{\n  \"query\": \"artificial intelligence breakthroughs 2025\",\n  \"count\": 5>\n</use_mcp_tool>"}
    ]
]

async def process_streaming_output(messages, client):
    """Process streaming output from the client and display events."""
    print("\n" + "="*80)
    print(f"TESTING WITH MESSAGE: {messages[-1]['content'][:50]}...")
    print("="*80)
    
    start_time = time.time()
    content_chunks = 0
    mcp_events = 0
    
    try:
        async for event_type, event_data in client.generate_chat_completion(
            messages=messages, 
            temperature=0.7,
            stream=True
        ):
            if event_type == "content":
                content = event_data.get("content", "")
                content_chunks += 1
                print(f"CONTENT[{content_chunks}]: {content}")
            
            elif event_type == "mcp_tool":
                mcp_events += 1
                content = event_data.get("content", "")
                print(f"MCP_TOOL[{mcp_events}]: {content[:100]}...")
            
            elif event_type == "error":
                print(f"ERROR: {event_data.get('error', 'Unknown error')}")
            
            elif event_type == "done":
                print("DONE")
    
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\nSUMMARY:")
    print(f"- Duration: {duration:.2f} seconds")
    print(f"- Content chunks: {content_chunks}")
    print(f"- MCP tool events: {mcp_events}")
    print("="*80 + "\n")

async def main():
    """Run the tests."""
    client = EnhancedCopilotClient()
    
    # Check if we have a valid token
    if not await client.ensure_valid_token():
        print("ERROR: Could not get a valid token. Please check your environment.")
        return
    
    print("\n\n===== ENHANCED STREAMING MCP TOOL TEST =====\n")
    
    # Process each test case
    for i, messages in enumerate(TEST_MESSAGES):
        print(f"\nTEST CASE {i+1}/{len(TEST_MESSAGES)}")
        await process_streaming_output(messages, client)
        
        # Wait a bit between tests
        if i < len(TEST_MESSAGES) - 1:
            print("Waiting 3 seconds before next test...")
            await asyncio.sleep(3)
    
    print("\nAll tests completed!\n")

if __name__ == "__main__":
    asyncio.run(main())
