"""Unit tests for the AI Horde OpenAI Interposer Layer."""

import pytest
from unittest.mock import MagicMock, patch

from horde_openai.translate import (
    convert_messages_to_prompt,
    translate_chat_request_to_horde,
    translate_horde_response_to_chat,
    estimate_tokens,
    _format_chatml,
    _format_mistral,
    _format_alpaca,
)
from horde_openai.models import ModelCapabilities, ModelRegistry
from horde_openai.client import (
    AIHordeClient,
    AIHordeError,
    InvalidRequestError,
    RequestTimeoutError,
)


class TestTranslate:
    """Tests for request/response translation."""

    def test_convert_messages_to_prompt_chatml(self):
        """Test converting messages to ChatML format."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        prompt = convert_messages_to_prompt(messages, format_name="ChatML")

        assert "### System:" in prompt
        assert "You are a helpful assistant." in prompt
        assert "### User:" in prompt
        assert "Hello, how are you?" in prompt
        assert "### Response:" in prompt

    def test_convert_messages_to_prompt_mistral(self):
        """Test converting messages to Mistral format."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        prompt = convert_messages_to_prompt(messages, format_name="Mistral")

        assert "<s>[INST]" in prompt
        assert "You are a helpful assistant." in prompt
        assert "Hello, how are you?" in prompt
        assert "[/INST]" in prompt

    def test_convert_messages_to_prompt_alpaca(self):
        """Test converting messages to Alpaca format."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        prompt = convert_messages_to_prompt(messages, format_name="Alpaca")

        assert "### Instruction:" in prompt
        assert "You are a helpful assistant." in prompt
        assert "Hello, how are you?" in prompt
        assert "### Response:" in prompt

    def test_convert_messages_to_prompt_user_only(self):
        """Test converting user-only messages."""
        messages = [
            {"role": "user", "content": "Just a simple question"},
        ]

        prompt = convert_messages_to_prompt(messages, format_name="ChatML")

        assert "### User:" in prompt
        assert "Just a simple question" in prompt
        assert "### Response:" in prompt

    def test_format_chatml(self):
        """Test ChatML formatting."""
        prompt = _format_chatml("You are a bot.", "Hello!")
        assert "### System:\nYou are a bot." in prompt
        assert "### User:\nHello!" in prompt
        assert "### Response:" in prompt

    def test_format_mistral(self):
        """Test Mistral formatting."""
        prompt = _format_mistral("You are a bot.", "Hello!")
        assert prompt.startswith("<s>[INST]")
        assert "You are a bot." in prompt
        assert "Hello!" in prompt
        assert prompt.endswith("[/INST]\n")

    def test_format_alpaca(self):
        """Test Alpaca formatting."""
        prompt = _format_alpaca("You are a bot.", "Hello!")
        assert "### Instruction:" in prompt
        assert "You are a bot." in prompt
        assert "Hello!" in prompt
        assert "### Response:" in prompt

    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "Hello, world! This is a test."  # ~10 chars
        tokens = estimate_tokens(text)
        assert tokens >= 2
        assert tokens <= 10  # Rough estimate

    def test_estimate_tokens_empty(self):
        """Test token estimation for empty string."""
        assert estimate_tokens("") == 0

    def test_translate_chat_request_to_horde(self):
        """Test translating OpenAI request to AI Horde format."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Test question"},
        ]

        # Mock model registry
        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.get_capabilities.return_value = ModelCapabilities(
            max_context_length=4096,
            max_generation_length=4096,
            instruct_format="ChatML",
            trusted=False,
        )

        horde_payload = translate_chat_request_to_horde(
            messages=messages,
            model="test/model",
            params={
                "temperature": 0.8,
                "max_tokens": 100,
                "top_p": 0.95,
            },
            model_registry=mock_registry,
        )

        assert "prompt" in horde_payload
        assert "params" in horde_payload
        assert horde_payload["models"] == ["test/model"]
        assert horde_payload["trusted_workers"] is False
        assert horde_payload["slow_workers"] is True

        # Check params
        assert horde_payload["params"]["max_length"] == 100
        assert horde_payload["params"]["max_context_length"] == 4096
        assert horde_payload["params"]["temperature"] == 0.8
        assert horde_payload["params"]["top_p"] == 0.95

    def test_translate_chat_request_respects_capabilities(self):
        """Test that request respects model capabilities."""
        messages = [{"role": "user", "content": "Test"}]

        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.get_capabilities.return_value = ModelCapabilities(
            max_context_length=2048,
            max_generation_length=500,
            instruct_format="ChatML",
            trusted=True,
        )

        horde_payload = translate_chat_request_to_horde(
            messages=messages,
            model="test/model",
            params={"max_tokens": 1000},  # More than max_generation_length
            model_registry=mock_registry,
        )

        # Should be capped at max_generation_length
        assert horde_payload["params"]["max_length"] <= 500

    def test_translate_horde_response_to_chat(self):
        """Test translating AI Horde response to OpenAI format."""
        generations = [
            {
                "text": "This is the generated response.",
                "internal_error": False,
                "truncated": False,
            }
        ]

        response = translate_horde_response_to_chat(
            generations=generations,
            model="test/model",
            original_prompt="Test prompt",
        )

        assert response["id"].startswith("chatcmpl-")
        assert response["object"] == "chat.completion"
        assert response["model"] == "test/model"
        assert len(response["choices"]) == 1
        assert response["choices"][0]["message"]["role"] == "assistant"
        assert response["choices"][0]["message"]["content"] == "This is the generated response."
        assert response["choices"][0]["finish_reason"] == "stop"

    def test_translate_horde_response_multiple_generations(self):
        """Test translating multiple generations."""
        generations = [
            {"text": "Response 1", "internal_error": False, "truncated": False},
            {"text": "Response 2", "internal_error": False, "truncated": False},
        ]

        response = translate_horde_response_to_chat(
            generations=generations,
            model="test/model",
            original_prompt="Test",
        )

        assert len(response["choices"]) == 2
        assert response["choices"][0]["index"] == 0
        assert response["choices"][1]["index"] == 1

    def test_translate_horde_response_finish_reason_truncated(self):
        """Test finish reason for truncated response."""
        generations = [
            {"text": "Partial...", "internal_error": False, "truncated": True},
        ]

        response = translate_horde_response_to_chat(
            generations=generations,
            model="test/model",
            original_prompt="Test",
        )

        assert response["choices"][0]["finish_reason"] == "length"

    def test_translate_horde_response_finish_reason_error(self):
        """Test finish reason for error response."""
        generations = [
            {"text": "", "internal_error": True, "truncated": False},
        ]

        response = translate_horde_response_to_chat(
            generations=generations,
            model="test/model",
            original_prompt="Test",
        )

        assert response["choices"][0]["finish_reason"] == "error"


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_get_capabilities_defaults(self):
        """Test getting capabilities with defaults for unknown model."""
        registry = ModelRegistry()
        capabilities = registry.get_capabilities("unknown/model")

        assert capabilities.max_context_length == 2048
        assert capabilities.max_generation_length == 100
        assert capabilities.instruct_format == "ChatML"

    def test_get_capabilities_caches(self):
        """Test that capabilities are cached."""
        registry = ModelRegistry()
        caps1 = registry.get_capabilities("test")
        caps2 = registry.get_capabilities("test")
        # Check values are the same
        assert caps1.max_context_length == caps2.max_context_length
        assert caps1.max_generation_length == caps2.max_generation_length
        assert caps1.instruct_format == caps2.instruct_format

    def test_list_models_empty(self):
        """Test listing models when none are loaded."""
        registry = ModelRegistry()
        assert registry.list_models() == []

    def test_get_model_info_unknown(self):
        """Test getting model info for unknown model."""
        registry = ModelRegistry()
        info = registry.get_model_info("unknown")
        assert info is None


class TestAIHordeClient:
    """Tests for AIHordeClient."""

    def test_client_initialization(self):
        """Test client initialization with defaults."""
        client = AIHordeClient()

        assert client.api_key == "0000000000"
        assert client.base_url == "https://aihorde.net/api/v2"
        assert client.timeout == 120
        assert client.poll_interval == 2.0

    def test_client_custom_settings(self):
        """Test client initialization with custom settings."""
        client = AIHordeClient(
            api_key="custom_key",
            base_url="http://localhost:8000",
            timeout=60,
            poll_interval=1.0,
        )

        assert client.api_key == "custom_key"
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 60
        assert client.poll_interval == 1.0


class TestErrorHandling:
    """Tests for error handling."""

    def test_ai_horde_error(self):
        """Test AIHordeError exception."""
        error = AIHordeError("Test error", status_code=500)
        assert str(error) == "Test error"
        assert error.status_code == 500

    def test_request_timeout_error(self):
        """Test RequestTimeoutError exception."""
        error = RequestTimeoutError()
        assert "timed out" in str(error).lower()

    def test_request_timeout_error_custom_message(self):
        """Test RequestTimeoutError with custom message."""
        error = RequestTimeoutError("Custom timeout message")
        assert str(error) == "Custom timeout message"

    def test_invalid_request_error(self):
        """Test InvalidRequestError exception."""
        error = InvalidRequestError("Invalid parameters")
        assert str(error) == "Invalid parameters"
        assert error.status_code == 400


class TestServerModels:
    """Tests for server Pydantic models."""

    def test_chat_completion_request_defaults(self):
        """Test ChatCompletionRequest with default values."""
        from horde_openai.server import ChatCompletionRequest, ChatMessage

        request = ChatCompletionRequest(
            model="test/model",
            messages=[ChatMessage(role="user", content="Hello")],
        )

        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.max_tokens == 100
        assert request.n == 1
        assert request.stream is False
