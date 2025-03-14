# CopilotAPI

A simple API service for interacting with GitHub Copilot chat completions.

## Overview

CopilotAPI provides an OpenAI-compatible API endpoint for GitHub Copilot chat completions. It handles authentication, token refreshing, and API interactions with the Copilot service.

## Features

- OpenAI-compatible `/v1/chat/completions` endpoint
- Dynamic model specification (not hardcoded)
- Token management with automatic refresh
- Fallback to alternative models on rate limits
- Streaming and non-streaming responses
- Token counting for usage tracking

## Getting Started

### Prerequisites

- Python 3.8 or higher
- GitHub Copilot token

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

## Configuration

The following environment variables can be configured:

- `GITHUB_COPILOT_TOKEN`: Your GitHub Copilot token
- `DEFAULT_MODEL`: Default model to use (e.g., "claude-3.7-sonnet")
- `FALLBACK_MODEL`: Model to use when rate limited
- `LOG_LEVEL`: Logging level (e.g., "INFO", "DEBUG")
- `ALLOW_ORIGINS`: CORS allowed origins (comma-separated list)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
