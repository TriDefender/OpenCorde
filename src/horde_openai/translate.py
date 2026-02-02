"""Request and response translation between OpenAI and AI Horde formats."""

import time
import uuid
from typing import Any, Dict, List, Optional

from .models import ModelCapabilities, ModelRegistry


# Default capability values
DEFAULT_MAX_CONTEXT_LENGTH = 2048
DEFAULT_MAX_GENERATION_LENGTH = 100


class ChatMessage:
    """Represents a chat message in the OpenAI format."""

    def __init__(self, role: str, content: str, name: Optional[str] = None):
        self.role = role
        self.content = content
        self.name = name

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """Create from dictionary format."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            name=data.get("name"),
        )


def convert_messages_to_prompt(
    messages: List[Dict[str, Any]],
    format_name: str = "ChatML",
) -> str:
    """Convert OpenAI messages array to prompt string for AI Horde.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        format_name: Instruct format to use (ChatML, Mistral, Alpaca)

    Returns:
        Formatted prompt string
    """
    system_message = ""
    user_messages: List[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_message = content
        elif role == "user":
            user_messages.append(content)
        elif role == "assistant":
            # Add assistant response to prompt if present
            user_messages.append(f"{content}")

    # Combine user messages (typically just the last one matters for simple cases)
    if not user_messages:
        user_content = ""
    elif len(user_messages) == 1:
        user_content = user_messages[0]
    else:
        # Join multiple user messages with appropriate separators
        user_content = "\n\n".join(user_messages)

    return _format_prompt(system_message, user_content, format_name)


def _format_prompt(system: str, user: str, format_name: str) -> str:
    """Format prompt according to instruct format."""
    format_lower = format_name.lower()

    if "mistral" in format_lower:
        return _format_mistral(system, user)
    elif "alpaca" in format_lower:
        return _format_alpaca(system, user)
    else:
        # Default to ChatML
        return _format_chatml(system, user)


def _format_chatml(system: str, user: str) -> str:
    """Format in ChatML style."""
    parts = []

    if system:
        parts.append(f"### System:\n{system}")

    if user:
        parts.append(f"### User:\n{user}")

    parts.append("### Response:")
    return "\n\n".join(parts)


def _format_mistral(system: str, user: str) -> str:
    """Format in Mistral instruct style."""
    content = system

    if system and user:
        content += "\n\n"

    if user:
        content += f"{user}"

    if content:
        return f"<s>[INST] {content} [/INST]\n"
    else:
        return "<s>[INST]  [/INST]\n"


def _format_alpaca(system: str, user: str) -> str:
    """Format in Alpaca style."""
    parts = []

    if system:
        parts.append(f"### Instruction:\n{system}")
    else:
        parts.append("### Instruction:\n")

    if user:
        parts.append(user)

    parts.append("### Response:")
    return "\n\n".join(parts)


def translate_chat_request_to_horde(
    messages: List[Dict[str, Any]],
    model: str,
    params: Dict[str, Any],
    model_registry: ModelRegistry,
) -> Dict[str, Any]:
    """Translate OpenAI chat completion request to AI Horde async format.

    Args:
        messages: OpenAI message format
        model: Model name
        params: OpenAI parameters (temperature, max_tokens, etc.)
        model_registry: Registry for model capabilities

    Returns:
        AI Horde async request payload
    """
    capabilities = model_registry.get_capabilities(model)

    # Convert messages to prompt
    prompt = convert_messages_to_prompt(
        messages,
        format_name=capabilities.instruct_format,
    )

    # Map parameters
    max_tokens = params.get("max_tokens", 100)
    max_length = min(max_tokens, capabilities.max_generation_length)
    max_context = capabilities.max_context_length

    # Map penalties - use the maximum of frequency and presence penalty
    freq_penalty = params.get("frequency_penalty", 0.0)
    pres_penalty = params.get("presence_penalty", 0.0)
    rep_pen = max(freq_penalty, pres_penalty)
    if rep_pen == 0.0:
        rep_pen = 1.1  # Default value

    return {
        "prompt": prompt,
        "params": {
            "max_length": max_length,
            "max_context_length": max_context,
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9),
            "top_k": params.get("top_k", 40),
            "rep_pen": rep_pen,
            "n": params.get("n", 1),
        },
        "models": [model],
        "trusted_workers": capabilities.trusted,
        "slow_workers": True,  # Prefer slower workers for better availability
    }


def translate_horde_response_to_chat(
    generations: List[Dict[str, Any]],
    model: str,
    original_prompt: str,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Translate AI Horde async response to OpenAI chat completion format.

    Args:
        generations: List of generation results from AI Horde
        model: Model name used
        original_prompt: Original prompt for token counting
        request_id: Optional request ID (generated if not provided)

    Returns:
        OpenAI chat completion response format
    """
    if request_id is None:
        request_id = str(uuid.uuid4())

    created_time = int(time.time())

    # Calculate usage
    prompt_tokens = estimate_tokens(original_prompt)

    choices = []
    for idx, gen in enumerate(generations):
        text = gen.get("text", "")
        completion_tokens = estimate_tokens(text)

        choice = {
            "index": idx,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "finish_reason": _get_finish_reason(gen),
        }
        choices.append(choice)

    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": created_time,
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _get_finish_reason(gen: Dict[str, Any]) -> str:
    """Determine finish reason from AI Horde generation."""
    # Check for various error conditions
    internal_error = gen.get("internal_error", False)
    if internal_error:
        return "error"

    # Check for truncation
    truncated = gen.get("truncated", False)
    if truncated:
        return "length"

    # Default to stop
    return "stop"


def translate_horde_response_to_chat_stream(
    generations: List[Dict[str, Any]],
    model: str,
    original_prompt: str,
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Create streaming chunks from AI Horde response.

    Note: AI Horde doesn't support true streaming, so we simulate it
    by returning the full response as a single chunk.
    """
    response = translate_horde_response_to_chat(
        generations,
        model,
        original_prompt,
        request_id,
    )

    # Create a single chunk with the full response
    chunk = {
        "id": response["id"],
        "object": "chat.completion.chunk",
        "created": response["created"],
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": response["choices"][0]["message"]["content"],
                },
                "finish_reason": response["choices"][0]["finish_reason"],
            }
        ],
    }

    return [chunk]


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string.

    Uses simple character-based estimation (~4 chars per token).
    This is a rough approximation for usage stats.
    """
    if not text:
        return 0
    return len(text) // 4


def translate_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    code: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an OpenAI-style error response."""
    error = {
        "message": message,
        "type": error_type,
    }
    if code:
        error["code"] = code

    return {
        "object": "error",
        "message": message,
        "type": error_type,
        "param": None,
        "code": code,
    }
