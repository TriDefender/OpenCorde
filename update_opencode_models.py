#!/usr/bin/env python3
"""
AI Horde Model Updater for OpenCode

This script fetches available text generation models from AI Horde's
/v2/workers endpoint and updates opencode.json with the model list.

It runs continuously, refreshing the model list every 5 minutes.

Usage:
    python update_opencode_models.py [--output opencode.json] [--interval 300]
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import httpx

# AI Horde API configuration
AI_HORDE_BASE_URL = "https://aihorde.net/api/v2"
DEFAULT_OUTPUT_FILE = "opencode.json"
DEFAULT_INTERVAL = 300  # 5 minutes


def format_model_name(model_id: str) -> str:
    """Format model ID to a readable display name."""
    # Extract the model name after the slash
    if "/" in model_id:
        name = model_id.split("/")[-1]
    else:
        name = model_id

    # Clean up common suffixes
    for suffix in [".gguf", "-Q8_0", "-Q6_K", "-Q5_K_M", "-Q4_K_M", "-IQ3_XS", "-i1"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    return name


def get_capabilities_from_worker(worker: Dict) -> Dict:
    """Extract capabilities from a worker object."""
    return {
        "max_context_length": worker.get("max_context_length", 4096),
        "max_generation_length": min(worker.get("max_length", 4096), 4096),
    }


async def fetch_text_workers() -> List[Dict]:
    """Fetch all text generation workers from AI Horde."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{AI_HORDE_BASE_URL}/workers?type=text",
            headers={"Client-Agent": "horde-openai-model-updater:1.0"},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


async def get_available_models() -> Dict[str, Dict]:
    """Get available models with their worker counts and capabilities."""
    workers = await fetch_text_workers()

    models: Dict[str, Dict] = {}

    for worker in workers:
        worker_models = worker.get("models", [])
        worker_id = worker.get("id", "unknown")
        capabilities = get_capabilities_from_worker(worker)

        for model_name in worker_models:
            if model_name not in models:
                models[model_name] = {
                    "worker_count": 0,
                    "max_context_length": capabilities["max_context_length"],
                    "max_generation_length": capabilities["max_generation_length"],
                }

            models[model_name]["worker_count"] += 1
            # Use minimum values across workers
            models[model_name]["max_context_length"] = min(
                models[model_name]["max_context_length"], capabilities["max_context_length"]
            )
            models[model_name]["max_generation_length"] = min(
                models[model_name]["max_generation_length"], capabilities["max_generation_length"]
            )

    return models


def generate_opencode_config(models: Dict[str, Dict]) -> Dict:
    """Generate OpenCode provider configuration from models."""

    # Sort models by worker count (most available first)
    sorted_models = sorted(models.items(), key=lambda x: (-x[1]["worker_count"], x[0]))

    # Build the models configuration
    models_config: Dict[str, Dict] = {}

    for model_id, info in sorted_models:
        # Use a sanitized key for the model
        key = model_id.replace("/", "_").replace("-", "_").replace(".", "_")

        models_config[key] = {
            "id": model_id,
            "name": f"{format_model_name(model_id)} (AI Horde)",
            "limit": {
                "context": info["max_context_length"],
                "output": min(info["max_generation_length"], 4096),
            },
        }

    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "ai-horde": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "AI Horde",
                "options": {
                    "baseURL": "http://localhost:8080/v1",
                },
                "models": models_config,
            }
        },
        "model": list(models_config.keys())[0] if models_config else None,
    }


def load_existing_config(path: str) -> Optional[Dict]:
    """Load existing opencode.json if it exists."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing config at {path}")
            return None
    return None


def merge_configs(existing: Dict, new: Dict) -> Dict:
    """Merge new config with existing config, preserving other settings."""
    if existing is None:
        return new

    result = existing.copy()

    if "provider" in new:
        result["provider"] = result.get("provider", {})
        result["provider"]["ai-horde"] = new["provider"]["ai-horde"]

    if new.get("model"):
        result["model"] = new["model"]

    if new.get("$schema"):
        result["$schema"] = new["$schema"]

    return result


def save_config(config: Dict, path: str):
    """Save configuration to file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Saved config to {path}")


async def update_models(output_file: str):
    """Update opencode.json with latest models from AI Horde."""
    print(f"[{datetime.now().isoformat()}] Fetching models from AI Horde...")

    try:
        models = await get_available_models()
        print(f"Found {len(models)} unique models")

        # Generate new config
        new_config = generate_opencode_config(models)

        # Load and merge with existing config
        existing = load_existing_config(output_file)
        merged = merge_configs(existing, new_config)

        # Save updated config
        save_config(merged, output_file)

        return True
    except httpx.HTTPError as e:
        print(f"Error fetching models: {e}")
        return False
    except Exception as e:
        print(f"Error updating models: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Update opencode.json with AI Horde models")
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output file path (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Refresh interval in seconds (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument("--once", "-1", action="store_true", help="Run once and exit (don't loop)")

    args = parser.parse_args()

    print(f"AI Horde Model Updater for OpenCode")
    print(f"Output file: {args.output}")
    print(f"Refresh interval: {args.interval} seconds")
    print("-" * 50)

    # Handle signals for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        print("\nShutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initial update
    success = await update_models(args.output)

    if args.once:
        return 0 if success else 1

    # Continuous update loop
    while not shutdown_event.is_set():
        # Wait for interval or shutdown
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=args.interval)
        except asyncio.TimeoutError:
            # Timeout means interval passed, time for next update
            await update_models(args.output)

    print("Done.")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
