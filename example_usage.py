#!/usr/bin/env python3
"""
Example usage of AI Horde OpenAI API Interposer

This script demonstrates how to use the interposer layer to make
OpenAI-compatible requests to AI Horde.
"""

import asyncio
import json
from horde_openai import AIHordeClient, create_app
from horde_openai.translate import convert_messages_to_prompt


async def example_direct_client():
    """Example: Using the client directly."""
    print("=" * 60)
    print("Example 1: Direct Client Usage")
    print("=" * 60)

    async with AIHordeClient() as client:
        # Refresh model registry
        await client.refresh_model_registry()

        # List available models
        models = client.list_models()
        print(f"\nAvailable models ({len(models)} total):")
        for model in models[:5]:  # Show first 5
            info = client.get_model_info(model)
            if info:
                caps = info.get("capabilities", {})
                print(f"  - {model}")
                print(f"    Context: {caps.get('max_context_length', 'N/A')} tokens")
                print(f"    Max gen: {caps.get('max_generation_length', 'N/A')} tokens")

        # Create a chat completion
        print("\n" + "-" * 40)
        print("Creating chat completion...")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Once upon a time in a magical forest,"},
        ]

        # The client handles translation and polling automatically
        response = await client.chat_completion(
            messages=messages,
            model="koboldcpp/LLaMA2-13B-Psyfighter2",
            max_tokens=50,
            temperature=0.7,
        )

        print("\nResponse:")
        print(json.dumps(response, indent=2))


async def example_translation():
    """Example: Understanding the translation."""
    print("\n" + "=" * 60)
    print("Example 2: Request Translation")
    print("=" * 60)

    # OpenAI request format
    openai_request = {
        "model": "koboldcpp/LLaMA2-13B-Psyfighter2",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short story about a dragon."},
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }

    print("\nOpenAI Request:")
    print(json.dumps(openai_request, indent=2))

    # Convert to prompt (simulating what the translate function does)
    prompt = convert_messages_to_prompt(openai_request["messages"], format_name="ChatML")

    print("\nConverted AI Horde Prompt:")
    print(prompt)


async def main():
    """Run all examples."""
    print("AI Horde OpenAI API Interposer - Example Usage")
    print("This demonstrates how to use the interposer layer")
    print("to make OpenAI-compatible requests to AI Horde.")
    print()

    # Run examples
    await example_translation()

    # Uncomment to run live example (requires network access)
    # await example_direct_client()

    print("\n" + "=" * 60)
    print("To run the server:")
    print("  uvicorn horde_openai.server:app --host 0.0.0.0 --port 8080")
    print("\nThen make requests to:")
    print("  POST http://localhost:8080/v1/chat/completions")
    print("  GET http://localhost:8080/v1/models")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
