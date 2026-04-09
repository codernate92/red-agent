"""Mistral AI target via the ``mistralai`` SDK."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from .base_target import BaseTarget, ConversationMessage, TargetError, TargetResponse

logger = logging.getLogger(__name__)


class MistralTarget(BaseTarget):
    """Mistral API target (mistral-large, mistral-small, codestral, ...)."""

    provider: ClassVar[str] = "mistral"

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str = "MISTRAL_API_KEY",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        try:
            from mistralai.client import Mistral as _Mistral
            import mistralai as _mistralai_module
        except ImportError as exc:  # pragma: no cover
            raise TargetError(
                "mistralai package is required for MistralTarget. "
                "Install with: pip install mistralai"
            ) from exc

        api_key = self._require_env(api_key_env)
        self._client = _Mistral(api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens

        transient: list[type[BaseException]] = list(self.RETRIABLE_EXCEPTIONS)
        for name in ("SDKError", "HTTPValidationError"):
            cls = getattr(_mistralai_module, name, None)
            if cls is not None:
                transient.append(cls)
        self.RETRIABLE_EXCEPTIONS = tuple(transient)  # type: ignore[misc]

    async def _aquery(
        self,
        messages: list[ConversationMessage],
        **kwargs: Any,
    ) -> TargetResponse:
        payload = [{"role": m.role, "content": m.content} for m in messages]

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "temperature": kwargs.pop("temperature", self.temperature),
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = kwargs.pop("max_tokens", self.max_tokens)
        request_kwargs.update(kwargs)

        # mistralai >=1.0 exposes async methods under client.chat.complete_async.
        response = await self._client.chat.complete_async(**request_kwargs)

        choice = response.choices[0]
        message = choice.message
        text = getattr(message, "content", "") or ""
        if isinstance(text, list):
            # Content can be a list of content chunks in newer SDKs.
            text = "".join(
                getattr(chunk, "text", "") or "" for chunk in text
            )

        usage = getattr(response, "usage", None)

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
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )
