"""Google Gemini target via the ``google-genai`` SDK.

Gemini's API uses ``generate_content()`` with a ``contents`` list rather than
OpenAI-style chat completions. We map our provider-agnostic
``ConversationMessage`` list to Gemini's shape and promote the most recent
``system`` message to ``system_instruction``.

Safety ratings and ``block_reason`` are captured in `metadata` so downstream
refusal detection / evaluators can see why Gemini blocked a response.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from .base_target import BaseTarget, ConversationMessage, TargetError, TargetResponse

logger = logging.getLogger(__name__)


class GoogleTarget(BaseTarget):
    """Google Gemini target (Gemini 2.0 Flash, 2.5 Pro, ...)."""

    provider: ClassVar[str] = "google"

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str = "GOOGLE_API_KEY",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:  # pragma: no cover
            raise TargetError(
                "google-genai package is required for GoogleTarget. "
                "Install with: pip install google-genai"
            ) from exc

        api_key = self._require_env(api_key_env)
        self._genai = genai
        self._types = genai_types
        self._client = genai.Client(api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Google exceptions live under google.genai.errors in recent SDKs.
        transient: list[type[BaseException]] = list(self.RETRIABLE_EXCEPTIONS)
        try:
            from google.genai import errors as genai_errors
            for name in ("APIError", "ServerError", "ClientError"):
                cls = getattr(genai_errors, name, None)
                if cls is not None:
                    transient.append(cls)
        except ImportError:
            pass
        self.RETRIABLE_EXCEPTIONS = tuple(transient)  # type: ignore[misc]

    async def _aquery(
        self,
        messages: list[ConversationMessage],
        **kwargs: Any,
    ) -> TargetResponse:
        system_parts: list[str] = []
        contents: list[Any] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            # Gemini roles: "user" and "model" (not "assistant").
            role = "user" if m.role == "user" else "model"
            contents.append(
                self._types.Content(
                    role=role,
                    parts=[self._types.Part.from_text(text=m.content)],
                )
            )

        config_kwargs: dict[str, Any] = {
            "temperature": kwargs.pop("temperature", self.temperature),
        }
        if self.max_tokens is not None:
            config_kwargs["max_output_tokens"] = kwargs.pop(
                "max_tokens", self.max_tokens
            )
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)

        config = self._types.GenerateContentConfig(**config_kwargs)

        # The google-genai SDK exposes an async surface at client.aio.
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        text = getattr(response, "text", None) or ""

        # Token counts
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        completion_tokens = (
            getattr(usage, "candidates_token_count", None) if usage else None
        )
        total_tokens = getattr(usage, "total_token_count", None) if usage else None

        # Safety + block information — preserved as metadata so the evaluator
        # can treat hard-blocked responses as refusals.
        md: dict[str, Any] = {}
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is not None:
            block_reason = getattr(prompt_feedback, "block_reason", None)
            if block_reason is not None:
                md["block_reason"] = str(block_reason)

        finish_reason: str | None = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = str(getattr(candidates[0], "finish_reason", None) or "")
            safety_ratings = getattr(candidates[0], "safety_ratings", None)
            if safety_ratings:
                md["safety_ratings"] = [
                    {
                        "category": str(getattr(r, "category", "")),
                        "probability": str(getattr(r, "probability", "")),
                    }
                    for r in safety_ratings
                ]

        return TargetResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            metadata=md,
        )

    def _should_retry_exception(self, exc: BaseException) -> bool:
        message = str(exc).lower()
        hard_fail_markers = (
            "resource_exhausted",
            "quota exceeded",
            "limit: 0",
            "api key not valid",
        )
        return not any(marker in message for marker in hard_fail_markers)
