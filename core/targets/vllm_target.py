"""Local open-weight target served via vLLM's OpenAI-compatible API."""

from __future__ import annotations

import logging
import os
from typing import Any, ClassVar

from .openai_target import OpenAITarget
from .base_target import TargetError

logger = logging.getLogger(__name__)


class VLLMTarget(OpenAITarget):
    """Thin wrapper over `OpenAITarget` pointing at a local vLLM server.

    vLLM exposes an OpenAI-compatible `/v1/chat/completions` endpoint, so we
    reuse the OpenAI client with a custom `base_url`. The API key is usually
    unused but required by the SDK; we default to the literal "EMPTY" value
    that vLLM accepts unless an env var is explicitly set.
    """

    provider: ClassVar[str] = "vllm"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key_env: str = "VLLM_API_KEY",
        **kwargs: Any,
    ) -> None:
        resolved_base_url = (
            base_url
            or os.environ.get("VLLM_BASE_URL")
            or "http://localhost:8000/v1"
        )
        # vLLM typically does not require auth. If the env var is unset,
        # inject a placeholder key so the OpenAI SDK is satisfied — without
        # ever hard-coding real credentials.
        if not os.environ.get(api_key_env):
            os.environ[api_key_env] = "EMPTY"

        try:
            super().__init__(
                model,
                api_key_env=api_key_env,
                base_url=resolved_base_url,
                **kwargs,
            )
        except TargetError:
            raise
        logger.info("vLLM target initialized: model=%s base_url=%s",
                    model, resolved_base_url)
