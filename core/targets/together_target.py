"""Together AI target — OpenAI-compatible, reuses :class:`OpenAITarget`.

Together provides API access to the open-weight Llama, Qwen, Gemma, Mixtral,
and DeepSeek families used in the FAR.AI scaling study, so this target is the
primary entry point for cross-family scaling comparisons without local GPUs.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from .openai_target import OpenAITarget

logger = logging.getLogger(__name__)


class TogetherTarget(OpenAITarget):
    """Together AI target.

    Together's endpoint speaks the OpenAI protocol, so we inherit `OpenAITarget`
    and just swap the `base_url` and default `api_key_env`. Model identifiers
    are Together-native strings like
    ``meta-llama/Llama-3.1-8B-Instruct-Turbo``.
    """

    provider: ClassVar[str] = "together"
    DEFAULT_BASE_URL: ClassVar[str] = "https://api.together.xyz/v1"

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str = "TOGETHER_API_KEY",
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            api_key_env=api_key_env,
            base_url=base_url or self.DEFAULT_BASE_URL,
            **kwargs,
        )
