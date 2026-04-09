"""Ollama target — OpenAI-compatible local endpoint for quick testing."""

from __future__ import annotations

import logging
import os
from typing import Any, ClassVar

from .openai_target import OpenAITarget

logger = logging.getLogger(__name__)


class OllamaTarget(OpenAITarget):
    """Ollama target.

    Ollama exposes an OpenAI-compatible ``/v1`` endpoint at
    ``http://localhost:11434/v1`` by default and requires no authentication,
    so we inject a placeholder API key into the environment when none exists.
    Model identifiers are Ollama tags like ``llama3.1:8b`` or ``qwen2.5:7b``.
    """

    provider: ClassVar[str] = "ollama"
    DEFAULT_BASE_URL: ClassVar[str] = "http://localhost:11434/v1"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key_env: str = "OLLAMA_API_KEY",
        **kwargs: Any,
    ) -> None:
        if not os.environ.get(api_key_env):
            os.environ[api_key_env] = "ollama"
        super().__init__(
            model,
            api_key_env=api_key_env,
            base_url=base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or self.DEFAULT_BASE_URL,
            **kwargs,
        )
