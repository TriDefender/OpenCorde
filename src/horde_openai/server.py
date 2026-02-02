"""FastAPI server with OpenAI-compatible endpoints."""

import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .client import (
    AIHordeClient,
    AIHordeError,
    InvalidRequestError,
    RequestTimeoutError,
)
from .models import ModelRegistry
from .translate import (
    translate_chat_request_to_horde,
    translate_error_response,
    translate_horde_response_to_chat,
    translate_horde_response_to_chat_stream,
)


# Global client instance
_client: Optional[AIHordeClient] = None


def get_client() -> AIHordeClient:
    """Get or create the global client instance."""
    global _client
    if _client is None:
        _client = AIHordeClient()
    return _client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _client
    _client = AIHordeClient()
    # Pre-populate model registry
    try:
        await _client.refresh_model_registry()
    except Exception:
        pass
    yield
    await _client.close()
    _client = None


# Create FastAPI application
app = FastAPI(
    title="AI Horde OpenAI API Interposer",
    description="OpenAI-compatible API for AI Horde distributed computing",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""

    role: str = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    """Chat completion request in OpenAI format."""

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 100
    n: Optional[int] = 1
    stream: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None


class ModelInfo(BaseModel):
    """Model information in OpenAI format."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "ai-horde"
    permission: List = Field(default_factory=list)
    root: Optional[str] = None
    parent: Optional[str] = None
    capabilities: Dict[str, Any] = Field(default_factory=dict)


class ModelListResponse(BaseModel):
    """Model list response in OpenAI format."""

    object: str = "list"
    data: List[ModelInfo]


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceInfo(BaseModel):
    """Choice information in chat completion."""

    index: int
    message: Dict[str, str]
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """Chat completion response in OpenAI format."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChoiceInfo]
    usage: UsageInfo


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models from AI Horde."""
    client = get_client()

    try:
        # Refresh model registry
        await client.refresh_model_registry()

        model_list = client.list_models()
        models = []

        for model_name in model_list:
            model_info = client.get_model_info(model_name)
            if model_info:
                models.append(ModelInfo(**model_info))

        return ModelListResponse(data=models)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion.

    Translates OpenAI request to AI Horde async format,
    submits to AI Horde, polls for completion, and returns
    OpenAI-compatible response.
    """
    client = get_client()

    # Convert messages to dict format
    messages = [msg.model_dump() for msg in request.messages]

    # Check if model is available
    capabilities = client.get_model_capabilities(request.model)
    if not capabilities.online:
        return JSONResponse(
            status_code=400,
            content=translate_error_response(
                f"Model {request.model} is not currently available",
                "invalid_request_error",
                "model_not_found",
            ),
        )

    try:
        # Translate request to AI Horde format
        horde_payload = translate_chat_request_to_horde(
            messages=messages,
            model=request.model,
            params={
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
                "n": request.n,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stop": request.stop,
            },
            model_registry=client.model_registry,
        )

        # Store original prompt for token estimation
        original_prompt = horde_payload["prompt"]

        # Submit and wait for completion
        generations = await client.submit_and_wait(horde_payload)

        # Translate response to OpenAI format
        if request.stream:
            # AI Horde doesn't support true streaming - return full response as single chunk
            chunks = translate_horde_response_to_chat_stream(
                generations=generations,
                model=request.model,
                original_prompt=original_prompt,
            )

            async def generate_stream():
                import json

                for chunk in chunks:
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
            )
        else:
            response = translate_horde_response_to_chat(
                generations=generations,
                model=request.model,
                original_prompt=original_prompt,
            )
            return JSONResponse(content=response)

    except InvalidRequestError as e:
        return JSONResponse(
            status_code=400,
            content=translate_error_response(str(e), "invalid_request_error", "invalid_request"),
        )
    except RequestTimeoutError as e:
        return JSONResponse(
            status_code=504,
            content=translate_error_response(str(e), "timeout", "request_timeout"),
        )
    except AIHordeError as e:
        return JSONResponse(
            status_code=502,
            content=translate_error_response(str(e), "service_unavailable", "horde_error"),
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI Horde OpenAI API Interposer",
        "version": "0.1.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
        },
    }


def create_app() -> FastAPI:
    """Create and return the FastAPI application.

    This is the main entry point for the server.
    """
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "horde_openai.server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )
