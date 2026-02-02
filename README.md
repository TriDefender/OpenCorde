# AI Horde OpenAI API Interposer

An OpenAI-compatible API layer for [AI Horde](https://aihorde.net) that enables opencode and any OpenAI-compatible client to use AI Horde's distributed GPU network for text generation.

## Overview

AI Horde is a crowdsourced distributed cluster of text and image generation workers. This interposer translates OpenAI chat completion requests to AI Horde's native async format and handles the polling workflow.

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  opencode   │───▶│  Interposer      │───▶│  AI Horde   │
│  or any     │     │  (this project)  │     │  API        │
│  OpenAI SDK │◀───│                  │◀───│             │
└─────────────┘     └──────────────────┘     └─────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Model Registry│
                  │ (from workers)│
                  └───────────────┘
```

## Features

- **OpenAI-compatible endpoints**: `/v1/chat/completions` and `/v1/models`
- **Automatic request translation**: Converts OpenAI format to AI Horde format
- **Async polling**: Handles submit/poll/retrieve workflow automatically
- **Model discovery**: Fetches capabilities from `/v2/workers?type=text`
- **Instruct format support**: ChatML, Mistral, and Alpaca prompt formats
- **OpenCode integration**: Auto-updating `opencode.json` with available models

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Start the Server

```bash
uvicorn horde_openai.server:app --host 0.0.0.0 --port 8080
```

### 2. Make a Request

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "koboldcpp/Fimbulvetr-11B-v2",
    "messages": [{"role": "user", "content": "Hello! How are you?"}],
    "max_tokens": 50
  }'
```

### 3. List Models

```bash
curl http://localhost:8080/v1/models
```

## OpenCode Integration

### Generate opencode.json

```bash
python update_opencode_models.py --once
```

This creates an `opencode.json` with:
- All available AI Horde text models
- Proper OpenCode provider format
- Model-specific context/output limits
- Default model set to most available

### Keep Models Updated

```bash
# Run continuously with 5-minute refresh (default)
python update_opencode_models.py

# Custom refresh interval
python update_opencode_models.py --interval 600
```

## Project Structure

```
HordeStreaming/
├── src/horde_openai/
│   ├── __init__.py       # Package exports
│   ├── client.py         # AI Horde HTTP client with async polling
│   ├── models.py         # Model registry from /v2/workers
│   ├── translate.py      # Request/response translation
│   └── server.py         # FastAPI server with OpenAI endpoints
├── tests/
│   └── test_interposer.py # 26 unit tests
├── docs/
│   └── INTERPOSER_SPEC.md # API specification
├── opencode.json         # OpenCode provider config (auto-generated)
├── pyproject.toml        # Package configuration
└── update_opencode_models.py  # Model updater script
```

## API Reference

### Chat Completions

**POST** `/v1/chat/completions`

```json
{
  "model": "koboldcpp/Fimbulvetr-11B-v2",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Tell me a joke."}
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
```

### List Models

**GET** `/v1/models`

Returns all available text generation models with their capabilities.

## How It Works

1. **Request received** at `/v1/chat/completions`
2. **Translate** OpenAI messages to AI Horde prompt format
3. **Submit** to `/api/v2/generate/text/async` → get job ID
4. **Poll** `/api/v2/generate/text/status/{id}` until done
5. **Translate** response back to OpenAI format
6. **Return** completion response

## Available Models

Currently available models (varies by worker availability):

- `koboldcpp/Fimbulvetr-11B-v2`
- `koboldcpp/NeonMaid-12B-v2`
- `TheDrummer/Cydonia-24B-v4.3`
- `koboldcpp/L3-8B-Stheno-v3.2`
- `zai-org/GLM-4.6`
- And 24+ more...

Run `python update_opencode_models.py --once` to refresh the model list.

## Testing

```bash
pytest tests/ -v
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_HORDE_API_KEY` | AI Horde API key | `0000000000` (anonymous) |

Even though you can use the anon apikey, still, its suggested for you to [obtain your key](https://aihorde.net/register) (and contribute back to the horde).

## Limitations

- **No true streaming**: AI Horde's async API doesn't support real-time streaming
- **Latency**: 2-30 seconds depending on queue
- **Token limit**: Maximum 4096 tokens per generation
- **Availability**: Depends on volunteer workers

## License

MIT
