"""
red-agent harness — Campaign execution engine and adaptive attack selection.

Re-exports:
    CampaignRunner: Main campaign execution engine.
    AdaptiveSelector: Adaptive probe selection based on results.
"""

from .runner import CampaignRunner
from .adaptive import AdaptiveSelector

__all__ = [
    "CampaignRunner",
    "AdaptiveSelector",
]
