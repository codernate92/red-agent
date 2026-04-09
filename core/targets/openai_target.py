"""OpenAI chat-completions target."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from .base_target import BaseTarget, ConversationMessage, TargetError, TargetResponse

logger = logging.getLogger(__name__)


class OpenAITarget(BaseTarget):
    """OpenAI Chat Completions target (GPT-4o, GPT-4o-mini, ...).

    Uses the async OpenAI SDK. API keys are read from an environment variable
    (default `OPENAI_API_KEY`); they are never accepted in code.
    """

    provider: ClassVar[str] = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        try:
            from openai import AsyncOpenAI
            import openai as _openai_module
        except ImportError as exc:  # pragma: no cover - dependency check
            raise TargetError(
                "openai package is required for OpenAITarget. "
                "Install with: pip install openai"
            ) from exc

        api_key = self._require_env(api_key_env)
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.request_timeout,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Extend retriable exceptions with OpenAI-specific transient errors.
        transient: list[type[BaseException]] = list(self.RETRIABLE_EXCEPTIONS)
        for name in ("APIConnectionError", "APITimeoutError", "RateLimitError",
                     "InternalServerError"):
            exc_cls = getattr(_openai_module, name, None)
            if exc_cls is not None:
                transient.append(exc_cls)
        self.RETRIABLE_EXCEPTIONS = tuple(transient)  # type: ignore[misc]

    async def _aquery(
        self,
        messages: list[ConversationMessage],
        **kwargs: Any,
    ) -> TargetResponse:
        payload_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        completion = await self._client.chat.completions.create(
            model=self.model,
            messages=payload_messages,
            temperature=kwargs.pop("temperature", self.temperature),
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            **kwargs,
        )

        choice = completion.choices[0]
        text = choice.message.content or ""
        usage = getattr(completion, "usage", None)

        return TargetResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=(
                getattr(usage, "completion_tokens", None) if usage else None
            ),
            total_tokens=getattr(usage, "total_tokens", None) if usage else None,
            finish_reason=getattr(choice, "finish_reason", None),
            raw=completion.model_dump() if hasattr(completion, "model_dump") else {},
        )
