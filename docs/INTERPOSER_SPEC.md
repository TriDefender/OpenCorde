# AI Horde OpenAI API Interposer Layer

This document describes the AI Horde Interposer Layer that translates opencode AI SDK requests
to AI Horde's native async API format and vice versa.

## Architecture

```
opencode (AI SDK) → Interposer Layer → AI Horde API
                      ↓
              Model Registry
              (from /v2/workers)
```

## API Endpoints

### 1. Chat Completions (OpenAI-compatible)

**POST** `/v1/chat/completions`

Translates OpenAI chat completion requests to AI Horde async text generation.

**Request (OpenAI format):**
```json
{
  "model": "koboldcpp/LLaMA2-13B-Psyfighter2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Once upon a time in a magical forest,"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": false
}
```

**Translation to AI Horde format:**
```json
{
  "prompt": "### System:\nYou are a helpful assistant.\n\n### User:\nOnce upon a time in a magical forest,\n\n### Response:",
  "params": {
    "max_length": 100,
    "max_context_length": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "rep_pen": 1.1,
    "n": 1
  },
  "models": ["koboldcpp/LLaMA2-13B-Psyfighter2"],
  "trusted_workers": false,
  "slow_workers": true
}
```

### 2. List Models

**GET** `/v1/models`

Lists available models with their capabilities from `/v2/workers`.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "koboldcpp/LLaMA2-13B-Psyfighter2",
      "object": "model",
      "created": 1700000000,
      "owned_by": "ai-horde",
      "permission": [],
      "root": "koboldcpp/LLaMA2-13B-Psyfighter2",
      "parent": null,
      "capabilities": {
        "max_context_length": 4096,
        "max_generation_length": 4096,
        "parameters": 13000000000,
        "instruct_format": "ChatML"
      }
    }
  ]
}
```

## Request Translation Rules

### OpenAI → AI Horde

| OpenAI Field | AI Horde Field | Notes |
|-------------|----------------|-------|
| `messages` | `prompt` | Convert messages array to prompt string using instruct format |
| `temperature` | `params.temperature` | Direct mapping |
| `max_tokens` | `params.max_length` | Capped at 4096 |
| `top_p` | `params.top_p` | Direct mapping |
| `model` | `models` | Wrap in array |
| `frequency_penalty` | `params.rep_pen` | Approximate mapping |
| `presence_penalty` | `params.rep_pen` | Approximate mapping |

### Instruct Format Conversion

| Instruct Format | Template |
|----------------|----------|
| ChatML | `### System:\n{system}\n\n### User:\n{user}\n\n### Response:` |
| Mistral | `<s>[INST] {system}\n\n{user} [/INST]\n` |
| Alpaca | `### Instruction:\n{system}\n{user}\n\n### Response:` |

## Model Capabilities from /v2/workers

The `/v2/workers` endpoint returns worker information that can be aggregated to determine model capabilities:

```json
[
  {
    "id": "worker-uuid",
    "name": "MyGPU",
    "type": "text",
    "online": true,
    "models": ["koboldcpp/LLaMA2-13B-Psyfighter2"],
    "max_length": 4096,
    "max_context_length": 4096,
    "trusted": true
  }
]
```

**Aggregation Logic:**
- `max_context_length`: Minimum of all workers' max_context_length for a model
- `max_generation_length`: Minimum of all workers' max_length (capped at 4096)
- `parameters`: From model reference database
- `instruct_format`: From model reference database (db.json)

## Async Workflow

### Step 1: Submit Request
```python
response = requests.post(
    "https://aihorde.net/api/v2/generate/text/async",
    headers={"apikey": API_KEY, "Client-Agent": "interposer:1.0"},
    json=translated_payload
)
job_id = response.json()["id"]
```

### Step 2: Poll for Completion
```python
while True:
    status = requests.get(
        f"https://aihorde.net/api/v2/generate/text/status/{job_id}",
        headers={"Client-Agent": "interposer:1.0"}
    )
    data = status.json()
    if data["done"]:
        return data["generations"]
    time.sleep(2)
```

### Step 3: Translate Response
```python
openai_response = {
    "id": f"chatcmpl-{job_id}",
    "object": "chat.completion",
    "created": int(time.time()),
    "model": original_model,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generation["text"]
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": count_tokens(original_prompt),
        "completion_tokens": count_tokens(generation["text"]),
        "total_tokens": sum
    }
}
```

## Limitations

1. **Token Limit**: Maximum generation length is capped at 4096 tokens
2. **No Streaming**: AI Horde's async API doesn't support real-time streaming
3. **Latency**: Async workflow adds 2-30 seconds of latency depending on queue
4. **Parameters**: Not all OpenAI parameters are supported
5. **Models**: Only models available on AI Horde can be used

## Example Usage

```python
import requests
import time

API_KEY = "0000000000"  # Anonymous or your API key
BASE_URL = "http://localhost:8080"  # Interposer URL

def chat_completion(messages, model, max_tokens=100, temperature=0.7):
    # Build prompt from messages
    prompt = messages_to_prompt(messages)
    
    # Submit to interposer
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    )
    
    return response.json()

def list_models():
    response = requests.get(f"{BASE_URL}/v1/models")
    return response.json()
```
