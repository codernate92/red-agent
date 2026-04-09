"""Factory for constructing `BaseTarget` instances from config dicts or CLI strings."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .base_target import BaseTarget, TargetError
from .openai_target import OpenAITarget
from .anthropic_target import AnthropicTarget
from .google_target import GoogleTarget
from .mistral_target import MistralTarget
from .together_target import TogetherTarget
from .ollama_target import OllamaTarget
from .vllm_target import VLLMTarget


Provider = Literal[
    "openai", "anthropic", "google", "mistral", "together", "ollama", "vllm"
]


class TargetConfig(BaseModel):
    """Declarative target configuration.

    Example:
        TargetConfig(provider="openai", model="gpt-4o-mini")
        TargetConfig(provider="anthropic", model="claude-3-5-sonnet-latest")
        TargetConfig(provider="vllm", model="meta-llama/Llama-3.1-8B-Instruct",
                     base_url="http://localhost:8000/v1")
    """

    provider: Provider
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    max_retries: int = 5
    rate_limit_rps: float | None = None
    trajectory_log_path: str | None = None
    request_timeout: float = 60.0
    metadata: dict[str, Any] = Field(default_factory=dict)


PROVIDER_REGISTRY: dict[str, type[BaseTarget]] = {
    "openai": OpenAITarget,
    "anthropic": AnthropicTarget,
    "google": GoogleTarget,
    "mistral": MistralTarget,
    "together": TogetherTarget,
    "ollama": OllamaTarget,
    "vllm": VLLMTarget,
}

# Backward-compat alias.
_PROVIDER_REGISTRY = PROVIDER_REGISTRY


def create_target(config: TargetConfig | dict[str, Any]) -> BaseTarget:
    """Instantiate a `BaseTarget` from a config dict or `TargetConfig`.

    Raises:
        TargetError: if the provider is unknown or required env vars missing.
    """
    if isinstance(config, dict):
        config = TargetConfig(**config)

    cls = PROVIDER_REGISTRY.get(config.provider)
    if cls is None:
        raise TargetError(
            f"Unknown provider {config.provider!r}; "
            f"expected one of {sorted(PROVIDER_REGISTRY)}"
        )

    kwargs: dict[str, Any] = {
        "system_prompt": config.system_prompt,
        "temperature": config.temperature,
        "max_retries": config.max_retries,
        "rate_limit_rps": config.rate_limit_rps,
        "trajectory_log_path": config.trajectory_log_path,
        "request_timeout": config.request_timeout,
        "metadata": dict(config.metadata),
    }
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    if config.base_url is not None:
        kwargs["base_url"] = config.base_url
    if config.api_key_env is not None:
        kwargs["api_key_env"] = config.api_key_env

    return cls(config.model, **kwargs)


def parse_target_spec(spec: str) -> TargetConfig:
    """Parse a CLI target spec like ``openai:gpt-4o-mini`` into a `TargetConfig`.

    Accepted forms:
        ``<provider>:<model>``          provider is openai | anthropic | vllm
        ``<provider>:<model>@<url>``    optional base_url override (vLLM)

    Raises:
        TargetError: if the spec is malformed or the provider is unknown.
    """
    if ":" not in spec:
        raise TargetError(
            f"Invalid target spec {spec!r}; expected '<provider>:<model>'"
        )
    provider, _, rest = spec.partition(":")
    provider = provider.strip().lower()
    if provider not in PROVIDER_REGISTRY:
        raise TargetError(
            f"Unknown provider {provider!r}; "
            f"expected one of {sorted(PROVIDER_REGISTRY)}"
        )

    base_url: str | None = None
    model = rest
    if "@" in rest:
        model, _, base_url = rest.partition("@")

    model = model.strip()
    if not model:
        raise TargetError(f"Invalid target spec {spec!r}; model is empty")

    return TargetConfig(
        provider=provider,  # type: ignore[arg-type]
        model=model,
        base_url=base_url.strip() if base_url else None,
    )
