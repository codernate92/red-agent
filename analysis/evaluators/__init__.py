"""StrongREJECT-compatible response evaluators for red-agent.

Evaluators score a single (prompt, response) pair and return an
:class:`EvaluationResult` with calibrated [0, 1] severity so that results are
directly comparable to FAR.AI's published numbers in "Data Poisoning in LLMs:
Jailbreak-Tuning and Scaling Trends" (Bowen et al. 2024, arXiv:2408.02946).
"""

from __future__ import annotations

from .evaluator_base import (
    BaseEvaluator,
    EvaluationResult,
    evaluate_campaign,
)
from .strongreject import StrongREJECTEvaluator
from .sentiment_evaluator import SentimentEvaluator
from .code_backdoor_evaluator import CodeBackdoorEvaluator

# Backward-compatible short name used in some docs / examples.
StrongREJECT = StrongREJECTEvaluator

__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "evaluate_campaign",
    "StrongREJECT",
    "StrongREJECTEvaluator",
    "SentimentEvaluator",
    "CodeBackdoorEvaluator",
]
