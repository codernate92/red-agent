"""Registry of known models used across red-agent experiments.

The registry maps short, human-friendly aliases (e.g. ``llama-3.1-70b``) to a
`ModelSpec` describing the provider, native model ID, optional parameter count
in billions (used for scaling regressions), and model family (used for
fixed-effects grouping in the scaling analyzer).

`params_b` is `None` for closed models where the exact parameter count is
unknown; those points are excluded from scaling regressions but still
included in cross-provider comparisons.

Use :func:`resolve` to convert a registry alias or raw ``provider:model``
spec into a :class:`core.targets.target_factory.TargetConfig`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .targets.target_factory import TargetConfig, parse_target_spec


class ModelSpec(BaseModel):
    """Declarative metadata for a known model."""

    provider: str
    model: str
    params_b: float | None = None
    family: str
    notes: str | None = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Parameter counts come from public releases / model cards. For mixture-of-
# experts models we record total parameters (not active), which is what the
# FAR.AI scaling plots use as the x-axis.

MODEL_REGISTRY: dict[str, ModelSpec] = {
    # ---- OpenAI (closed) ----
    "gpt-4o": ModelSpec(
        provider="openai", model="gpt-4o", params_b=None, family="gpt"
    ),
    "gpt-4o-mini": ModelSpec(
        provider="openai", model="gpt-4o-mini", params_b=None, family="gpt"
    ),
    "gpt-4-turbo": ModelSpec(
        provider="openai", model="gpt-4-turbo", params_b=None, family="gpt"
    ),

    # ---- GPT-OSS (OpenAI open-weight, via Together) ----
    "gpt-oss-120b": ModelSpec(
        provider="together",
        model="openai/gpt-oss-120b",
        params_b=117,  # ~117B total MoE params
        family="gpt-oss",
        notes="OpenAI open-weight MoE (~5.1B active).",
    ),

    # ---- Anthropic (closed) ----
    "claude-sonnet-4": ModelSpec(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        params_b=None,
        family="claude",
    ),
    "claude-sonnet-4-6": ModelSpec(
        provider="anthropic",
        model="claude-sonnet-4-6",
        params_b=None,
        family="claude",
    ),
    "claude-sonnet-4-5": ModelSpec(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        params_b=None,
        family="claude",
    ),
    "claude-haiku-4-5": ModelSpec(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        params_b=None,
        family="claude",
    ),
    "claude-opus-4-6": ModelSpec(
        provider="anthropic",
        model="claude-opus-4-6",
        params_b=None,
        family="claude",
    ),

    # ---- Google (closed) ----
    "gemini-2.5-pro": ModelSpec(
        provider="google",
        model="gemini-2.5-pro-preview-05-06",
        params_b=None,
        family="gemini",
    ),
    "gemini-2.0-flash": ModelSpec(
        provider="google",
        model="gemini-2.0-flash",
        params_b=None,
        family="gemini",
    ),

    # ---- Llama 3.1 (open, via Together) ----
    "llama-3.1-8b": ModelSpec(
        provider="together",
        model="meta-llama/Llama-3.1-8B-Instruct-Turbo",
        params_b=8,
        family="llama-3.1",
    ),
    "llama-3.1-70b": ModelSpec(
        provider="together",
        model="meta-llama/Llama-3.1-70B-Instruct-Turbo",
        params_b=70,
        family="llama-3.1",
    ),
    "llama-3.1-405b": ModelSpec(
        provider="together",
        model="meta-llama/Llama-3.1-405B-Instruct-Turbo",
        params_b=405,
        family="llama-3.1",
    ),

    # ---- Gemma 2 (open, via Together) ----
    "gemma-2-9b": ModelSpec(
        provider="together",
        model="google/gemma-2-9b-it",
        params_b=9,
        family="gemma-2",
    ),
    "gemma-2-27b": ModelSpec(
        provider="together",
        model="google/gemma-2-27b-it",
        params_b=27,
        family="gemma-2",
    ),

    # ---- Mixtral (open, via Together) ----
    "mixtral-8x22b": ModelSpec(
        provider="together",
        model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        params_b=141,  # total params
        family="mixtral",
    ),

    # ---- Mistral (direct API) ----
    "mistral-small": ModelSpec(
        provider="mistral",
        model="mistral-small-latest",
        params_b=22,
        family="mistral",
    ),
    "mistral-large": ModelSpec(
        provider="mistral",
        model="mistral-large-latest",
        params_b=123,
        family="mistral",
    ),
    "mistral-nemo": ModelSpec(
        provider="together",
        model="mistralai/Mistral-Nemo-Instruct-2407",
        params_b=12,
        family="mistral",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_spec(alias: str) -> ModelSpec:
    """Look up a `ModelSpec` by alias. Raises KeyError if unknown."""
    return MODEL_REGISTRY[alias]


def list_aliases() -> list[str]:
    """Return all registered aliases sorted by family then size."""
    def sort_key(alias: str) -> tuple[str, float]:
        spec = MODEL_REGISTRY[alias]
        return (spec.family, spec.params_b or 0.0)
    return sorted(MODEL_REGISTRY, key=sort_key)


def by_family(family: str) -> list[str]:
    """Return registry aliases belonging to a given family (e.g. ``llama-3.1``)."""
    return [
        alias
        for alias, spec in MODEL_REGISTRY.items()
        if spec.family == family
    ]


def resolve(spec: str, **overrides: Any) -> TargetConfig:
    """Resolve a registry alias OR a raw ``provider:model`` spec to a config.

    Registry lookup is tried first; if it misses, the string is parsed as a
    raw provider:model spec. Any keyword arguments are merged into the
    returned ``TargetConfig`` (e.g. to set ``trajectory_log_path``).
    """
    if spec in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[spec]
        cfg = TargetConfig(
            provider=entry.provider,  # type: ignore[arg-type]
            model=entry.model,
            metadata={
                "alias": spec,
                "family": entry.family,
                "params_b": entry.params_b,
            },
        )
    else:
        cfg = parse_target_spec(spec)

    if overrides:
        cfg = cfg.model_copy(update=overrides)
    return cfg
