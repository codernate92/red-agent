"""Anthropic messages-API target."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from .base_target import BaseTarget, ConversationMessage, TargetError, TargetResponse

logger = logging.getLogger(__name__)


class AnthropicTarget(BaseTarget):
    """Anthropic Messages API target (Claude family).

    The Messages API takes the system prompt as a top-level parameter rather
    than as a `system` role message, so we strip system messages out of the
    conversation and pass the most recent one explicitly.
    """

    provider: ClassVar[str] = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        try:
            from anthropic import AsyncAnthropic
            import anthropic as _anthropic_module
        except ImportError as exc:  # pragma: no cover - dependency check
            raise TargetError(
                "anthropic package is required for AnthropicTarget. "
                "Install with: pip install anthropic"
            ) from exc

        api_key = self._require_env(api_key_env)
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=self.request_timeout,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

        transient: list[type[BaseException]] = list(self.RETRIABLE_EXCEPTIONS)
        for name in ("APIConnectionError", "APITimeoutError", "RateLimitError",
                     "InternalServerError"):
            exc_cls = getattr(_anthropic_module, name, None)
            if exc_cls is not None:
                transient.append(exc_cls)
        self.RETRIABLE_EXCEPTIONS = tuple(transient)  # type: ignore[misc]

    async def _aquery(
        self,
        messages: list[ConversationMessage],
        **kwargs: Any,
    ) -> TargetResponse:
        system_parts: list[str] = []
        turns: list[dict[str, str]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                turns.append({"role": m.role, "content": m.content})

        system_prompt = "\n\n".join(system_parts) if system_parts else None

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", self.temperature),
        }
        if system_prompt is not None:
            create_kwargs["system"] = system_prompt
        create_kwargs.update(kwargs)

        message = await self._client.messages.create(**create_kwargs)

        # Concatenate all text blocks in the response content.
        text_chunks: list[str] = []
        for block in message.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_chunks.append(getattr(block, "text", ""))
        text = "".join(text_chunks)

        usage = getattr(message, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", None) if usage else None
        completion_tokens = getattr(usage, "output_tokens", None) if usage else None
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        return TargetResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=getattr(message, "stop_reason", None),
            raw=message.model_dump() if hasattr(message, "model_dump") else {},
        )
