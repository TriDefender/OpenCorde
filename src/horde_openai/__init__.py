"""AI Horde OpenAI API Interposer Layer

This package provides OpenAI-compatible endpoints that translate requests
to AI Horde's native async API format.
"""

__version__ = "0.1.0"

from .client import AIHordeClient
from .models import ModelRegistry
from .server import create_app

__all__ = [
    "AIHordeClient",
    "ModelRegistry",
    "create_app",
]
