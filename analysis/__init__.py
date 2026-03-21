"""
red-agent analysis — Vulnerability scoring, attack surface mapping, and reporting.

Re-exports:
    VulnerabilityRating: Severity rating enum.
    AgentVulnerability: Individual vulnerability dataclass.
    VulnerabilityScorer: Scores vulnerabilities from campaign results.
    AttackSurfaceMap: Maps the agent's attack surface.
    RedTeamReport: Full red-team report generator.
"""

from .vulnerability import AgentVulnerability, VulnerabilityRating, VulnerabilityScorer
from .attack_surface import AttackSurfaceMap
from .report import RedTeamReport

__all__ = [
    "VulnerabilityRating",
    "AgentVulnerability",
    "VulnerabilityScorer",
    "AttackSurfaceMap",
    "RedTeamReport",
]
