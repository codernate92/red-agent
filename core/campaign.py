"""
Orchestrated attack campaigns for red-teaming LLM agents.

A campaign is a structured sequence of phases, each containing multiple
probes. Campaigns support different escalation strategies and produce
consolidated results with risk scoring.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .probe import Probe, ProbeResult, ProbeStatus

logger = logging.getLogger(__name__)


@dataclass
class CampaignPhase:
    """A phase within a red-team campaign.

    Each phase groups related probes and defines execution policy for that
    group (e.g. whether to continue on failure, minimum success threshold).

    Attributes:
        name: Phase name (e.g. ``"reconnaissance"``, ``"exploitation"``).
        description: What this phase aims to accomplish.
        probes: The probes to execute in this phase.
        continue_on_failure: If ``True``, keep running probes even if some fail.
        success_threshold: Fraction of probes that must succeed (0.0-1.0) to
            consider the phase successful and advance to the next one.
    """

    name: str
    description: str
    probes: list[Probe] = field(default_factory=list)
    continue_on_failure: bool = True
    success_threshold: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "probes": [p.to_dict() for p in self.probes],
            "continue_on_failure": self.continue_on_failure,
            "success_threshold": self.success_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignPhase:
        """Deserialize from a plain dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            probes=[Probe.from_dict(p) for p in data.get("probes", [])],
            continue_on_failure=data.get("continue_on_failure", True),
            success_threshold=data.get("success_threshold", 0.0),
        )


class EscalationStrategy(Enum):
    """Strategy for deciding which probes to run next based on results."""

    LINEAR = "linear"
    """Run all probes in defined order regardless of results."""

    ADAPTIVE = "adaptive"
    """Skip or add probes dynamically based on observed results."""

    DEPTH_FIRST = "depth_first"
    """When a technique succeeds, prioritize deeper exploitation of that vector."""

    BREADTH_FIRST = "breadth_first"
    """Try all techniques at each tactic level before advancing to the next."""


@dataclass
class CampaignConfig:
    """Configuration for a red-team campaign.

    Attributes:
        name: Campaign identifier.
        description: What this campaign tests.
        target_name: Name of the target agent under evaluation.
        phases: Ordered list of campaign phases.
        escalation_strategy: How to adapt probe execution based on results.
        max_total_probes: Hard cap on total probes executed.
        timeout_per_probe: Timeout for each probe in seconds.
        cool_down_between_probes: Delay between probes in seconds (rate-limit avoidance).
        record_all_exchanges: Whether to capture full request/response pairs.
    """

    name: str
    description: str
    target_name: str
    phases: list[CampaignPhase] = field(default_factory=list)
    escalation_strategy: EscalationStrategy = EscalationStrategy.LINEAR
    max_total_probes: int = 100
    timeout_per_probe: float = 30.0
    cool_down_between_probes: float = 0.5
    record_all_exchanges: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "target_name": self.target_name,
            "phases": [p.to_dict() for p in self.phases],
            "escalation_strategy": self.escalation_strategy.value,
            "max_total_probes": self.max_total_probes,
            "timeout_per_probe": self.timeout_per_probe,
            "cool_down_between_probes": self.cool_down_between_probes,
            "record_all_exchanges": self.record_all_exchanges,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignConfig:
        """Deserialize from a plain dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            target_name=data["target_name"],
            phases=[CampaignPhase.from_dict(p) for p in data.get("phases", [])],
            escalation_strategy=EscalationStrategy(
                data.get("escalation_strategy", "linear")
            ),
            max_total_probes=data.get("max_total_probes", 100),
            timeout_per_probe=data.get("timeout_per_probe", 30.0),
            cool_down_between_probes=data.get("cool_down_between_probes", 0.5),
            record_all_exchanges=data.get("record_all_exchanges", True),
        )


@dataclass
class CampaignResult:
    """Consolidated results from a completed red-team campaign.

    Attributes:
        campaign_name: Name of the campaign.
        target_name: Name of the target agent.
        start_time: Campaign start time (epoch seconds).
        end_time: Campaign end time (epoch seconds).
        phase_results: Map of phase name to list of probe results.
        successful_techniques: Technique IDs that succeeded at least once.
        failed_techniques: Technique IDs that never succeeded.
        attack_chain: Techniques that succeeded in sequence, representing
            a viable multi-step attack path.
        overall_risk_score: Aggregate risk score from 0.0 (no risk) to 10.0.
        summary: Human-readable summary of findings.
    """

    campaign_name: str
    target_name: str
    start_time: float
    end_time: float
    phase_results: dict[str, list[ProbeResult]] = field(default_factory=dict)
    successful_techniques: list[str] = field(default_factory=list)
    failed_techniques: list[str] = field(default_factory=list)
    attack_chain: list[str] = field(default_factory=list)
    overall_risk_score: float = 0.0
    summary: str = ""

    @property
    def duration_seconds(self) -> float:
        """Total campaign duration in seconds."""
        return self.end_time - self.start_time

    @property
    def total_probes(self) -> int:
        """Total number of probes executed across all phases."""
        return sum(len(results) for results in self.phase_results.values())

    @property
    def total_successes(self) -> int:
        """Number of probes that fully succeeded."""
        return sum(
            1
            for results in self.phase_results.values()
            for r in results
            if r.status == ProbeStatus.SUCCESS
        )

    @property
    def success_rate(self) -> float:
        """Fraction of probes that succeeded."""
        total = self.total_probes
        if total == 0:
            return 0.0
        return self.total_successes / total

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "campaign_name": self.campaign_name,
            "target_name": self.target_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "phase_results": {
                phase: [r.to_dict() for r in results]
                for phase, results in self.phase_results.items()
            },
            "successful_techniques": list(self.successful_techniques),
            "failed_techniques": list(self.failed_techniques),
            "attack_chain": list(self.attack_chain),
            "overall_risk_score": self.overall_risk_score,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignResult:
        """Deserialize from a plain dictionary."""
        phase_results: dict[str, list[ProbeResult]] = {}
        for phase, results_data in data.get("phase_results", {}).items():
            phase_results[phase] = [ProbeResult.from_dict(r) for r in results_data]

        return cls(
            campaign_name=data["campaign_name"],
            target_name=data["target_name"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            phase_results=phase_results,
            successful_techniques=data.get("successful_techniques", []),
            failed_techniques=data.get("failed_techniques", []),
            attack_chain=data.get("attack_chain", []),
            overall_risk_score=data.get("overall_risk_score", 0.0),
            summary=data.get("summary", ""),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string.

        Args:
            indent: JSON indentation level.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> CampaignResult:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(json_str))
