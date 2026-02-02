"""Model registry for AI Horde models with capabilities from /v2/workers."""

import asyncio
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


AI_HORDE_BASE_URL = "https://aihorde.net/api/v2"


@dataclass
class ModelCapabilities:
    """Model capabilities from AI Horde workers."""

    max_context_length: int = 2048
    max_generation_length: int = 100
    parameters: Optional[int] = None
    instruct_format: str = "ChatML"
    online: bool = True
    trusted: bool = False


class ModelRegistry:
    """Registry of available models with their capabilities.

    Caches model capabilities from /v2/workers with a configurable TTL.
    """

    def __init__(
        self,
        api_key: str = "0000000000",
        base_url: str = AI_HORDE_BASE_URL,
        cache_ttl: int = 300,  # 5 minutes
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.cache_ttl = cache_ttl
        self._capabilities: Dict[str, ModelCapabilities] = {}
        self._last_refresh: float = 0
        self._refresh_task: Optional[asyncio.Task] = None
        self._model_reference_db: Dict[str, Dict[str, Any]] = {}

    async def refresh(self) -> None:
        """Refresh model capabilities from /v2/workers."""
        try:
            import httpx

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
                workers = response.json()

            # Aggregate capabilities by model
            model_workers: Dict[str, list] = {}
            for worker in workers:
                worker_models = worker.get("models", [])
                worker_id = worker.get("id", "unknown")
                worker_online = worker.get("online", False)
                worker_trusted = worker.get("trusted", False)
                max_length = min(worker.get("max_length", 4096), 4096)
                max_context = worker.get("max_context_length", 4096)

                for model_name in worker_models:
                    if model_name not in model_workers:
                        model_workers[model_name] = []
                    model_workers[model_name].append(
                        {
                            "id": worker_id,
                            "online": worker_online,
                            "trusted": worker_trusted,
                            "max_length": max_length,
                            "max_context": max_context,
                        }
                    )

            # Build capabilities from aggregated data
            new_capabilities: Dict[str, ModelCapabilities] = {}
            for model_name, workers_info in model_workers.items():
                if not workers_info:
                    continue

                # Check if any worker is online
                online = any(w["online"] for w in workers_info)
                trusted = any(w["trusted"] for w in workers_info)

                # Get minimum context length and max generation length across workers
                min_context = min(w["max_context"] for w in workers_info)
                max_gen = min(w["max_length"] for w in workers_info)

                # Get instruct format from model reference
                instruct_format = self._get_instruct_format(model_name)
                parameters = self._get_parameters(model_name)

                new_capabilities[model_name] = ModelCapabilities(
                    max_context_length=min_context,
                    max_generation_length=max_gen,
                    parameters=parameters,
                    instruct_format=instruct_format,
                    online=online,
                    trusted=trusted,
                )

            self._capabilities = new_capabilities
            self._last_refresh = time.time()

        except httpx.HTTPError as e:
            # On error, keep existing cache
            pass

    def _get_instruct_format(self, model_name: str) -> str:
        """Get instruct format for a model from reference database."""
        if model_name in self._model_reference_db:
            return self._model_reference_db[model_name].get("instruct_format", "ChatML")

        # Default format mappings based on model name patterns
        if "mistral" in model_name.lower() or "mixtral" in model_name.lower():
            return "Mistral"
        elif "alpaca" in model_name.lower() or "vicuna" in model_name.lower():
            return "Alpaca"
        elif "llama2" in model_name.lower() or "llama-2" in model_name.lower():
            return "ChatML"
        elif "qwen" in model_name.lower():
            return "ChatML"
        else:
            return "ChatML"

    def _get_parameters(self, model_name: str) -> Optional[int]:
        """Get approximate parameter count for a model."""
        if model_name in self._model_reference_db:
            return self._model_reference_db[model_name].get("parameters")

        # Common model parameter estimates based on name patterns
        if "7b" in model_name.lower():
            return 7_000_000_000
        elif "13b" in model_name.lower():
            return 13_000_000_000
        elif "30b" in model_name.lower():
            return 30_000_000_000
        elif "34b" in model_name.lower():
            return 34_000_000_000
        elif "65b" in model_name.lower():
            return 65_000_000_000
        elif "70b" in model_name.lower():
            return 70_000_000_000
        elif "405b" in model_name.lower():
            return 405_000_000_000
        elif "3b" in model_name.lower():
            return 3_000_000_000
        elif "1b" in model_name.lower():
            return 1_000_000_000
        else:
            return None

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a specific model.

        Triggers background refresh if cache is stale.
        """
        current_time = time.time()

        # Trigger background refresh if cache is stale
        if current_time - self._last_refresh > self.cache_ttl:
            try:
                if self._refresh_task is None or self._refresh_task.done():
                    self._refresh_task = asyncio.create_task(self.refresh())
            except RuntimeError:
                # No running event loop - skip background refresh
                pass

        # Return cached capabilities or default
        return self._capabilities.get(
            model_name,
            ModelCapabilities(
                max_context_length=2048,
                max_generation_length=100,
            ),
        )

    def list_models(self) -> list[str]:
        """List all available model names."""
        return list(self._capabilities.keys())

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get OpenAI-style model information."""
        capabilities = self.get_capabilities(model_name)

        if model_name not in self._capabilities:
            return None

        return {
            "id": model_name,
            "object": "model",
            "created": int(self._last_refresh),
            "owned_by": "ai-horde",
            "permission": [],
            "root": model_name,
            "parent": None,
            "capabilities": {
                "max_context_length": capabilities.max_context_length,
                "max_generation_length": capabilities.max_generation_length,
                "parameters": capabilities.parameters,
                "instruct_format": capabilities.instruct_format,
            },
        }

    async def ensure_fresh(self) -> None:
        """Ensure cache is fresh by waiting for refresh to complete."""
        current_time = time.time()

        if current_time - self._last_refresh > self.cache_ttl:
            await self.refresh()
        elif self._refresh_task is not None:
            # Wait for any pending refresh
            await self._refresh_task
