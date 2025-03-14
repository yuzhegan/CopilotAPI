"""Main application module for the CopilotAPI service."""

import json
import time
import asyncio
from typing import Dict, Any, List, Optional

import tiktoken
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from app.client.copilot import CopilotClient
from app.config.settings import get_settings
from app.utils.logger import logger
from app.utils.token import get_token_config_async, schedule_token_refresh
from app.utils.redis_manager import RedisManager

# Create FastAPI application
settings = get_settings()
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
)

# Initialize Redis manager
redis_manager = RedisManager()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tiktoken encoder for token counting
encoding = tiktoken.encoding_for_model("gpt-4")

# Background task for token refresh
token_refresh_task = None

@app.on_event("startup")
async def startup_event():
    """Execute startup tasks."""
    # Log Redis connection status
    if redis_manager.is_connected():
        logger.info("Redis connection established, using Redis for token storage")
    else:
        logger.warning("Redis connection failed, using memory storage only")
    
    # Initialize token on startup
    token_config = await get_token_config_async()
    if token_config:
        logger.info(f"Initial token loaded, expires at: {token_config.get('expires_at', 0)}")
    else:
        logger.warning("Failed to load initial token")
    
    # Start background token refresh task
    global token_refresh_task
    token_refresh_task = asyncio.create_task(schedule_token_refresh())
    logger.info("Token refresh background task started")

@app.on_event("shutdown")
async def shutdown_event():
    """Execute shutdown tasks."""
    # Cancel token refresh background task
    global token_refresh_task
    if token_refresh_task:
        token_refresh_task.cancel()
        try:
            await token_refresh_task
        except asyncio.CancelledError:
            logger.info("Token refresh task cancelled")

# Utility functions
def generate_chat_id() -> str:
    """Generate a unique chat ID."""
    return f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"

def create_response_chunk(
    chat_id: str,
    created_time: int,
    model: str,
    content: str = "",
    finish_reason: Optional[str] = None,
) -> bytes:
    """Create a standard format response chunk."""
    delta = {"role": "assistant"}
    
    if content:
        delta["content"] = content
    
    response = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }
        ],
    }
    
    return f"data: {json.dumps(response)}\n\n".encode("utf-8")

# Check for required API token
def verify_token():
    if not settings.github_copilot_token:
        logger.error("GitHub Copilot token is required but not provided")
        raise HTTPException(
            status_code=500,
            detail="GitHub Copilot token not configured. Please set GITHUB_COPILOT_TOKEN in your environment."
        )
    return True

# API routes
@app.get("/")
async def root():
    """Root endpoint."""
    token_status = "configured" if settings.github_copilot_token else "not configured"
    redis_status = "connected" if redis_manager.is_connected() else "disconnected"
    
    # Get token expiration info if available
    token_info = ""
    if settings.copilot_config:
        expires_at = settings.copilot_config.get("expires_at", 0)
        current_time = int(time.time())
        if expires_at > current_time:
            token_info = f", expires in {expires_at - current_time} seconds"
    
    return {
        "message": "Welcome to CopilotAPI", 
        "version": settings.api_version,
        "github_token_status": token_status + token_info,
        "redis_status": redis_status,
        "storage_type": "Redis" if redis_manager.is_connected() else "Memory"
    }

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "claude-3.7-sonnet",
                "object": "model",
                "created": 1678944000,
                "owned_by": "anthropic",
            },
            {
                "id": "gpt-4",
                "object": "model",
                "created": 1678944000,
                "owned_by": "openai",
            },
        ],
    }

@app.get("/v1/token/refresh")
async def refresh_token():
    """
    Manually refresh the token.
    Useful for testing token refresh functionality.
    """
    from app.utils.token import fetch_copilot_token
    
    token = await fetch_copilot_token()
    if token:
        expires_at = token.get("expires_at", 0)
        current_time = int(time.time())
        return {
            "status": "success",
            "expires_in": expires_at - current_time,
            "storage": "Redis" if redis_manager.is_connected() else "Memory"
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to refresh token")

@app.post("/v1/chat/completions", dependencies=[Depends(verify_token)])
async def chat_completions(request: Request):
    """
    Chat completions endpoint, compatible with OpenAI API format.
    
    Expected request body:
    - messages: List of message objects with role and content
    - model: Model name
    - stream: Whether to stream the response
    - temperature: Temperature
    - top_p: Top-p
    - presence_penalty: Presence penalty
    - frequency_penalty: Frequency penalty
    """
    try:
        # Parse request body
        body = await request.json()
        
        # Extract required fields
        messages = body.get("messages")
        if not messages:
            raise HTTPException(status_code=400, detail="Missing 'messages' field")
        
        # Get model name, defaulting to claude-3.7-sonnet if not specified
        model = body.get("model", settings.default_model)
        
        # Extract optional parameters
        stream = body.get("stream", True)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 1.0)
        presence_penalty = body.get("presence_penalty", 0.0)
        frequency_penalty = body.get("frequency_penalty", 0.0)
        
        # Create client with specified model and fallback
        client = CopilotClient(
            model=model,
            fallback_model=settings.fallback_model
        )
        
        # Generate response
        if stream:
            # Stream response
            return await stream_response(
                client, 
                messages, 
                model, 
                temperature, 
                top_p, 
                presence_penalty, 
                frequency_penalty
            )
        else:
            # Non-streaming response
            return await non_stream_response(
                client, 
                messages, 
                model, 
                temperature, 
                top_p, 
                presence_penalty, 
                frequency_penalty
            )
            
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def stream_response(
    client: CopilotClient,
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float
) -> StreamingResponse:
    """Generate a streaming response."""
    
    async def stream_generator():
        chat_id = generate_chat_id()
        created_time = int(time.time())
        
        try:
            async for event_type, event_data in client.generate_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stream=True
            ):
                if event_type == "content":
                    content = event_data.get("content", "")
                    yield create_response_chunk(chat_id, created_time, model, content=content)
                elif event_type == "error":
                    logger.error(f"Error in stream: {event_data.get('error', '')}")
                    yield create_response_chunk(chat_id, created_time, model, content="Error occurred during generation")
                    yield create_response_chunk(chat_id, created_time, model, finish_reason="error")
                    yield b"data: [DONE]\n\n"
                    return
                elif event_type == "done":
                    yield create_response_chunk(chat_id, created_time, model, finish_reason="stop")
                    yield b"data: [DONE]\n\n"
                    return
        except Exception as e:
            logger.error(f"Error in stream generator: {e}")
            yield create_response_chunk(chat_id, created_time, model, content="An error occurred")
            yield create_response_chunk(chat_id, created_time, model, finish_reason="error")
            yield b"data: [DONE]\n\n"
    
    return StreamingResponse(stream_generator(), media_type="text/event-stream")

async def non_stream_response(
    client: CopilotClient,
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float
) -> Dict[str, Any]:
    """Generate a non-streaming response."""
    
    chat_id = generate_chat_id()
    created_time = int(time.time())
    
    try:
        # Calculate prompt tokens
        input_text = "\n".join(msg.get("content", "") for msg in messages)
        input_tokens = encoding.encode(input_text)
        
        # Get complete response
        result = await client.complete_chat_without_streaming(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty
        )
        
        content = result.get("content", "")
        
        # Calculate completion tokens
        output_tokens = encoding.encode(content)
        
        # Return response in OpenAI format
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(input_tokens),
                "completion_tokens": len(output_tokens),
                "total_tokens": len(input_tokens) + len(output_tokens),
            },
        }
    except Exception as e:
        logger.error(f"Error in non-streaming response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
