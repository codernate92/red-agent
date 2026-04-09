"""Real LLM target backends for red-agent.

Provides a uniform async interface (`BaseTarget`) over OpenAI, Anthropic,
Google Gemini, Mistral, Together AI, Ollama, and self-hosted vLLM, with
retry, rate limiting, and JSONL trajectory logging.
"""

from __future__ import annotations

from .base_target import (
    BaseTarget,
    ConversationMessage,
    TargetError,
    TargetResponse,
)
from .openai_target import OpenAITarget
from .anthropic_target import AnthropicTarget
from .google_target import GoogleTarget
from .mistral_target import MistralTarget
from .together_target import TogetherTarget
from .ollama_target import OllamaTarget
from .vllm_target import VLLMTarget
from .target_factory import (
    TargetConfig,
    create_target,
    parse_target_spec,
    PROVIDER_REGISTRY,
)

__all__ = [
    "BaseTarget",
    "ConversationMessage",
    "TargetError",
    "TargetResponse",
    "OpenAITarget",
    "AnthropicTarget",
    "GoogleTarget",
    "MistralTarget",
    "TogetherTarget",
    "OllamaTarget",
    "VLLMTarget",
    "TargetConfig",
    "create_target",
    "parse_target_spec",
    "PROVIDER_REGISTRY",
]
