# CopilotAPI

A simple API service for interacting with GitHub Copilot chat completions.

## Overview

CopilotAPI provides an OpenAI-compatible API endpoint for GitHub Copilot chat completions. It handles authentication, token refreshing, and API interactions with the Copilot service.

## Features

- OpenAI-compatible `/v1/chat/completions` endpoint
- Dynamic model specification (not hardcoded)
- Token management with automatic refresh
- Redis-based token storage for reliability across restarts
- Proactive token refreshing 3 minutes before expiration
- Background task for continuous token monitoring
- Fallback to alternative models on rate limits
- Streaming and non-streaming responses
- Token counting for usage tracking

## Getting Started

### Prerequisites

- Python 3.8 or higher
- GitHub Copilot token
- Redis (optional, for token persistence)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/copilot-api.git
   cd copilot-api
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env to set your GitHub Copilot token and other settings
   ```

### Running the Service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

## API Usage

### Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3.7-sonnet",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "stream": true
  }'
```

### List Models

```bash
curl -X GET http://localhost:8000/v1/models
```

### Manually Refresh Token

```bash
curl -X GET http://localhost:8000/v1/token/refresh
```

## Configuration

The following environment variables can be configured:

### API Settings
- `API_TITLE`: The title of the API
- `API_DESCRIPTION`: Description of the API
- `API_VERSION`: Version number
- `ALLOW_ORIGINS`: CORS allowed origins (comma-separated list)

### GitHub Copilot Settings
- `GITHUB_COPILOT_TOKEN`: Your GitHub Copilot token (required)
- `DEFAULT_MODEL`: Default model to use (e.g., "claude-3.7-sonnet")
- `FALLBACK_MODEL`: Model to use when rate limited

### Redis Settings
- `REDIS_HOST`: Redis server hostname (default: "localhost")
- `REDIS_PORT`: Redis server port (default: 6379)
- `REDIS_DB`: Redis database number (default: 0)
- `REDIS_PASSWORD`: Redis password if required

### Logging Settings
- `LOG_LEVEL`: Logging level (e.g., "INFO", "DEBUG")
- `LOG_FILE`: Log file path (optional)

## Token Management Details

The application handles GitHub Copilot tokens with a sophisticated approach:

1. **Redis Storage**: Tokens are stored in Redis for persistence across application restarts. If Redis is unavailable, the application gracefully falls back to in-memory storage.

2. **Automatic Refreshing**: A background task continuously monitors token expiration and refreshes tokens 3 minutes before they expire, ensuring uninterrupted service.

3. **Startup Validation**: On application startup, the system attempts to load a valid token from Redis or fetch a new one if needed.

4. **Fallback Mechanism**: If token refresh fails, the system will retry with exponential backoff, and clearly log all token-related events.

5. **Manual Control**: For testing and administrative purposes, the `/v1/token/refresh` endpoint allows manual token refresh.

## Architecture

The system follows a modular design with clear separation of concerns:

- **Client Layer**: The `CopilotClient` class handles all communication with the GitHub Copilot API.
- **Token Management**: The token utilities handle token acquisition, storage, and monitoring.
- **API Layer**: FastAPI endpoints implement OpenAI-compatible interfaces.
- **Storage Layer**: Redis integration provides persistent token storage.

## Error Handling

The application implements comprehensive error handling:

1. **API Errors**: HTTP errors from the Copilot API are properly captured and returned to the client.
2. **Token Errors**: Authentication issues are detected and reported with clear error messages.
3. **Fallback Models**: If rate limits are encountered, the system can automatically retry with a fallback model.
4. **General Exceptions**: Unexpected errors are caught, logged, and returned with appropriate status codes.

## Performance Considerations

- **Asynchronous Design**: The application is built with async/await patterns throughout, ensuring high throughput.
- **Connection Pooling**: Redis connections are managed efficiently to avoid resource exhaustion.
- **Minimal Blocking**: Token operations are designed to minimize blocking, even during refresh operations.
- **Efficient Streaming**: Response streaming is optimized for low latency.

## Security Notes

- The GitHub Copilot token is sensitive and should be properly secured.
- The `.env` file should never be committed to source control.
- Redis should be configured with authentication if exposed beyond localhost.
- Consider implementing API key authentication for production deployments.

## Differences from DeepClaude_1

This project is a focused adaptation of the DeepClaude_1 project, with the following key enhancements:

1. Simplified architecture focusing solely on Copilot API integration
2. Improved async/await implementation for better performance
3. More robust token management with proactive refresh
4. Enhanced error handling and logging
5. Cleaner API design with OpenAI compatibility

## License

This project is licensed under the MIT License - see the LICENSE file for details.
