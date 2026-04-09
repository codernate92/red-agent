"""Abstract base class for real LLM targets.

All concrete targets (OpenAI, Anthropic, vLLM, ...) subclass `BaseTarget` and
implement `_aquery()`. `BaseTarget` handles:

* Single-turn `query()` and multi-turn `converse()` interfaces.
* Metadata capture (latency, token counts, model name, raw provider payload).
* Exponential-backoff retry with jitter on transient errors.
* Optional per-target request-per-second rate limiting.
* JSONL trajectory logging of every prompt/response pair.
* Adapter methods (`respond`, `reset`) so targets conform to the existing
  `core.target.TargetAgent` async protocol used by `harness.runner.CampaignRunner`.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Iterable, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ConversationMessage(BaseModel):
    """Single chat message in provider-agnostic form."""

    role: str  # "system" | "user" | "assistant"
    content: str


class TargetResponse(BaseModel):
    """Structured result from a single LLM query.

    Captures everything downstream analysis (StrongREJECT, scaling studies,
    trajectory logging) needs — including latency, token usage, and the raw
    provider payload for audit.
    """

    text: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float = 0.0
    finish_reason: str | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    raw: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class _AsyncRateLimiter:
    """Minimum-interval async rate limiter (token-bucket-lite).

    Enforces at most `rps` calls per second across concurrent callers using a
    single asyncio lock + monotonic timestamp of the last request.
    """

    def __init__(self, rps: float | None) -> None:
        self._min_interval = (1.0 / rps) if rps and rps > 0 else 0.0
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Base target
# ---------------------------------------------------------------------------


class TargetError(RuntimeError):
    """Raised when a target cannot fulfill a query after retries."""


class BaseTarget(abc.ABC):
    """Abstract async LLM target with retry, rate limit, and JSONL logging.

    Args:
        model: Provider-specific model name (e.g. "gpt-4o-mini").
        system_prompt: Optional default system prompt applied to every query.
        trajectory_log_path: If set, every query/response is appended as JSONL.
        max_retries: Max attempts before raising `TargetError`.
        base_delay: Initial backoff delay in seconds; doubles per attempt.
        max_delay: Upper bound on a single backoff sleep.
        rate_limit_rps: Optional rps cap (float). `None` disables limiting.
        metadata: Arbitrary dict attached to every response's metadata field.
        request_timeout: Per-request HTTP timeout (seconds).
    """

    provider: ClassVar[str] = "base"

    # Exception types considered transient and worth retrying.
    # Subclasses may extend this tuple; we keep the base permissive so that
    # network errors from any SDK bubble up through retry.
    RETRIABLE_EXCEPTIONS: ClassVar[tuple[type[BaseException], ...]] = (
        asyncio.TimeoutError,
        ConnectionError,
        TimeoutError,
    )

    def __init__(
        self,
        model: str,
        *,
        system_prompt: str | None = None,
        trajectory_log_path: str | os.PathLike[str] | None = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        rate_limit_rps: float | None = None,
        metadata: dict[str, Any] | None = None,
        request_timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.trajectory_log_path = (
            Path(trajectory_log_path) if trajectory_log_path else None
        )
        if self.trajectory_log_path is not None:
            self.trajectory_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._limiter = _AsyncRateLimiter(rate_limit_rps)
        self.metadata = dict(metadata or {})
        self.request_timeout = request_timeout
        self._conversation_history: list[ConversationMessage] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> TargetResponse:
        """Send a single-turn user prompt and return a `TargetResponse`."""
        effective_system = system_prompt or self.system_prompt
        messages: list[ConversationMessage] = []
        if effective_system:
            messages.append(
                ConversationMessage(role="system", content=effective_system)
            )
        messages.append(ConversationMessage(role="user", content=prompt))
        return await self.converse(messages, **kwargs)

    async def converse(
        self,
        messages: Sequence[ConversationMessage | dict[str, str]],
        **kwargs: Any,
    ) -> TargetResponse:
        """Send a full conversation and return the assistant's next turn."""
        normalized = [self._coerce_message(m) for m in messages]
        await self._limiter.acquire()
        start = time.perf_counter()
        response = await self._retrying_query(normalized, **kwargs)
        response.latency_ms = (time.perf_counter() - start) * 1000.0
        # Merge any instance-level metadata without clobbering per-call keys.
        for k, v in self.metadata.items():
            response.metadata.setdefault(k, v)
        self._log_trajectory(normalized, response)
        return response

    # ------------------------------------------------------------------
    # Adapter methods — conform to core.target.TargetAgent async protocol
    # so existing CampaignRunner can drive real targets unchanged.
    # ------------------------------------------------------------------

    async def respond(
        self,
        message: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Return assistant text for one turn, tracking running history.

        Compatible with `core.target.TargetAgent`. The runner passes its own
        `conversation_history`; we honor it rather than our internal buffer so
        that probe-level resets stay consistent.
        """
        messages: list[ConversationMessage] = []
        if self.system_prompt:
            messages.append(
                ConversationMessage(role="system", content=self.system_prompt)
            )
        for turn in conversation_history or []:
            messages.append(self._coerce_message(turn))
        messages.append(ConversationMessage(role="user", content=message))
        result = await self.converse(messages)
        return result.text

    async def reset(self) -> None:
        """Clear any internal conversation state."""
        self._conversation_history.clear()

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def _aquery(
        self,
        messages: list[ConversationMessage],
        **kwargs: Any,
    ) -> TargetResponse:
        """Issue a single provider request. Subclasses implement this.

        Implementations MUST populate `provider`, `model`, `text`, and SHOULD
        populate token counts, `finish_reason`, and `raw` where available.
        Latency is measured by the caller and written afterwards.
        """

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _retrying_query(
        self,
        messages: list[ConversationMessage],
        **kwargs: Any,
    ) -> TargetResponse:
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries):
            try:
                return await asyncio.wait_for(
                    self._aquery(messages, **kwargs),
                    timeout=self.request_timeout,
                )
            except self.RETRIABLE_EXCEPTIONS as exc:
                last_exc = exc
                delay = min(
                    self.max_delay,
                    self.base_delay * (2**attempt) + random.uniform(0.0, 0.25),
                )
                logger.warning(
                    "%s query failed (attempt %d/%d): %s — retrying in %.2fs",
                    self.provider,
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            except Exception as exc:
                # Non-retriable: fail fast with context.
                raise TargetError(
                    f"{self.provider}:{self.model} query failed: {exc}"
                ) from exc

        raise TargetError(
            f"{self.provider}:{self.model} query failed after "
            f"{self.max_retries} retries: {last_exc}"
        )

    def _log_trajectory(
        self,
        messages: Iterable[ConversationMessage],
        response: TargetResponse,
    ) -> None:
        if self.trajectory_log_path is None:
            return
        record = {
            "timestamp": response.timestamp,
            "provider": response.provider,
            "model": response.model,
            "messages": [m.model_dump() for m in messages],
            "response": response.text,
            "latency_ms": response.latency_ms,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "finish_reason": response.finish_reason,
            "metadata": response.metadata,
        }
        try:
            with self.trajectory_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error(
                "Failed to append trajectory log %s: %s",
                self.trajectory_log_path,
                exc,
            )

    @staticmethod
    def _coerce_message(
        m: ConversationMessage | dict[str, Any],
    ) -> ConversationMessage:
        if isinstance(m, ConversationMessage):
            return m
        return ConversationMessage(role=str(m["role"]), content=str(m["content"]))

    @staticmethod
    def _require_env(env_var: str) -> str:
        value = os.environ.get(env_var)
        if not value:
            raise TargetError(
                f"Environment variable {env_var!r} is not set; cannot authenticate."
            )
        return value
