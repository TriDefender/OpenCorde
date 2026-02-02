"""AI Horde HTTP client for async requests with polling."""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from .models import AI_HORDE_BASE_URL, ModelRegistry


class AIHordeError(Exception):
    """Base exception for AI Horde client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class RequestTimeoutError(AIHordeError):
    """Raised when async request polling times out."""

    def __init__(self, message: str = "Request timed out"):
        super().__init__(message)


class InvalidRequestError(AIHordeError):
    """Raised for invalid request parameters."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class AIHordeClient:
    """Client for interacting with AI Horde API.

    Handles async request submission and polling for completion.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = AI_HORDE_BASE_URL,
        timeout: int = 120,
        poll_interval: float = 2.0,
    ):
        # Read API key from environment or use provided/default
        if api_key is None:
            api_key = os.environ.get("AI_HORDE_API_KEY", "0000000000")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.model_registry = ModelRegistry(api_key=api_key, base_url=base_url)

    async def submit_async_request(
        self,
        payload: Dict[str, Any],
    ) -> str:
        """Submit an async request to AI Horde.

        Args:
            payload: Translated request payload for AI Horde

        Returns:
            Job ID for polling

        Raises:
            AIHordeError: On API error
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/generate/text/async",
                headers={
                    "apikey": self.api_key,
                    "Content-Type": "application/json",
                    "Client-Agent": "horde-openai:0.1.0",
                },
                json=payload,
                timeout=30.0,
            )

            if response.status_code == 400:
                error_data = response.json()
                raise InvalidRequestError(error_data.get("message", "Invalid request"))

            response.raise_for_status()

            data = response.json()
            return data["id"]

    async def check_job_status(
        self,
        job_id: str,
    ) -> Dict[str, Any]:
        """Check status of an async job.

        Args:
            job_id: The job ID to check

        Returns:
            Job status response including 'done' and 'generations'
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/generate/text/status/{job_id}",
                headers={
                    "Client-Agent": "horde-openai:0.1.0",
                },
                timeout=30.0,
            )

            response.raise_for_status()
            return response.json()

    async def poll_for_completion(
        self,
        job_id: str,
        timeout: Optional[int] = None,
        poll_interval: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Poll for job completion.

        Args:
            job_id: Job ID to poll
            timeout: Maximum time to wait in seconds (default: client's timeout)
            poll_interval: Polling interval in seconds (default: client's poll_interval)

        Returns:
            List of generation results

        Raises:
            RequestTimeoutError: If job doesn't complete within timeout
            AIHordeError: On API error
        """
        if timeout is None:
            timeout = self.timeout
        if poll_interval is None:
            poll_interval = self.poll_interval

        start_time = time.time()
        attempts = 0

        while True:
            elapsed = time.time() - start_time

            if elapsed >= timeout:
                raise RequestTimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

            status = await self.check_job_status(job_id)

            if status.get("done", False):
                return status.get("generations", [])

            # Exponential backoff for polling interval
            current_interval = min(poll_interval * (2 ** (attempts // 3)), 10.0)
            await asyncio.sleep(current_interval)
            attempts += 1

    async def submit_and_wait(
        self,
        payload: Dict[str, Any],
        timeout: Optional[int] = None,
        poll_interval: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Submit request and wait for completion.

        Convenience method combining submit_async_request and poll_for_completion.

        Args:
            payload: Request payload for AI Horde
            timeout: Maximum time to wait
            poll_interval: Polling interval

        Returns:
            List of generation results
        """
        job_id = await self.submit_async_request(payload)
        return await self.poll_for_completion(job_id, timeout, poll_interval)

    async def get_workers(self) -> List[Dict[str, Any]]:
        """Get list of active text workers.

        Returns:
            List of worker information from /v2/workers?type=text
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/workers?type=text",
                headers={
                    "apikey": self.api_key,
                    "Client-Agent": "horde-openai:0.1.0",
                },
                timeout=30.0,
            )

            response.raise_for_status()
            return response.json()

    async def refresh_model_registry(self) -> None:
        """Refresh the model registry from AI Horde workers."""
        await self.model_registry.refresh()

    def get_model_capabilities(self, model_name: str):
        """Get capabilities for a specific model."""
        return self.model_registry.get_capabilities(model_name)

    def list_models(self) -> List[str]:
        """List available model names."""
        return self.model_registry.list_models()

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get OpenAI-style model information."""
        return self.model_registry.get_model_info(model_name)

    async def close(self):
        """Close the client and any resources."""
        # httpx.AsyncClient handles cleanup in context manager
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
