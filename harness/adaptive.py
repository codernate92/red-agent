"""
Adaptive attack selection for red-team campaigns.

Dynamically selects the next probes to execute based on prior results,
supporting multiple escalation strategies: linear, adaptive, depth-first,
and breadth-first.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from core.campaign import EscalationStrategy
from core.probe import Probe, ProbeResult, ProbeStatus
from core.taxonomy import TacticCategory, TechniqueRegistry

logger = logging.getLogger(__name__)


class AdaptiveSelector:
    """Selects the next probes to run based on observed results and strategy.

    Maintains running state of which techniques have been tried, which
    succeeded, and which failed, then uses this to prioritize the next
    batch of probes according to the chosen escalation strategy.

    Args:
        taxonomy: Technique registry for looking up technique metadata.
        initial_results: Optional list of results to seed the selector.
    """

    def __init__(
        self,
        taxonomy: TechniqueRegistry,
        initial_results: list[ProbeResult] | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self._results: list[ProbeResult] = []
        self._succeeded_techniques: set[str] = set()
        self._failed_techniques: set[str] = set()
        self._tried_techniques: set[str] = set()
        self._tactic_attempts: dict[TacticCategory, int] = defaultdict(int)
        self._tactic_successes: dict[TacticCategory, int] = defaultdict(int)

        if initial_results:
            for result in initial_results:
                self.update(result)

    def update(self, result: ProbeResult) -> None:
        """Feed a new probe result into the selector state.

        Args:
            result: The probe result to incorporate.
        """
        self._results.append(result)
        self._tried_techniques.add(result.technique_id)

        try:
            technique = self.taxonomy.get(result.technique_id)
            self._tactic_attempts[technique.tactic] += 1
        except KeyError:
            pass

        if result.status == ProbeStatus.SUCCESS:
            self._succeeded_techniques.add(result.technique_id)
            try:
                technique = self.taxonomy.get(result.technique_id)
                self._tactic_successes[technique.tactic] += 1
            except KeyError:
                pass
        elif result.status == ProbeStatus.FAILED:
            self._failed_techniques.add(result.technique_id)

    def suggest_next_probes(
        self,
        available_probes: list[Probe],
        strategy: EscalationStrategy,
        max_probes: int = 5,
    ) -> list[Probe]:
        """Select the next probes to execute based on strategy and results.

        Args:
            available_probes: All probes that could be run next.
            strategy: The escalation strategy to apply.
            max_probes: Maximum number of probes to return.

        Returns:
            Ordered list of up to ``max_probes`` probes to execute.
        """
        if not available_probes:
            return []

        if strategy == EscalationStrategy.LINEAR:
            return self._linear_select(available_probes, max_probes)
        elif strategy == EscalationStrategy.ADAPTIVE:
            return self._adaptive_select(available_probes, max_probes)
        elif strategy == EscalationStrategy.DEPTH_FIRST:
            return self._depth_first_select(available_probes, max_probes)
        elif strategy == EscalationStrategy.BREADTH_FIRST:
            return self._breadth_first_select(available_probes, max_probes)
        else:
            return available_probes[:max_probes]

    def get_attack_surface_estimate(self) -> dict[str, Any]:
        """Estimate which tactic categories the target is vulnerable to.

        Based on results observed so far, estimates vulnerability per
        tactic category.

        Returns:
            Dict mapping tactic category names to vulnerability estimates.
        """
        surface: dict[str, Any] = {}

        for tactic in TacticCategory:
            attempts = self._tactic_attempts.get(tactic, 0)
            successes = self._tactic_successes.get(tactic, 0)

            if attempts == 0:
                vulnerability = "untested"
                score = 0.0
            elif successes == 0:
                vulnerability = "resistant"
                score = 0.0
            else:
                rate = successes / attempts
                if rate >= 0.7:
                    vulnerability = "highly_vulnerable"
                elif rate >= 0.4:
                    vulnerability = "vulnerable"
                elif rate >= 0.1:
                    vulnerability = "partially_vulnerable"
                else:
                    vulnerability = "mostly_resistant"
                score = rate

            surface[tactic.value] = {
                "vulnerability": vulnerability,
                "attempts": attempts,
                "successes": successes,
                "success_rate": score,
            }

        return surface

    def _compute_technique_priority(self, technique_id: str) -> float:
        """Compute a heuristic priority score for a technique.

        Higher scores mean higher priority. Considers:
        - Whether prerequisites have succeeded (bonus)
        - Whether similar techniques failed (penalty)
        - Technique severity (bonus for higher severity)
        - Detection difficulty (bonus for harder to detect)

        Args:
            technique_id: The technique to score.

        Returns:
            Priority score (higher = more important to try next).
        """
        try:
            technique = self.taxonomy.get(technique_id)
        except KeyError:
            return 0.5

        priority = 1.0

        # Bonus if prerequisites succeeded
        for prereq_id in technique.prerequisites:
            if prereq_id in self._succeeded_techniques:
                priority += 2.0
            elif prereq_id in self._failed_techniques:
                priority -= 1.0

        # Severity weight
        severity_bonus = {
            "critical": 2.0,
            "high": 1.5,
            "medium": 1.0,
            "low": 0.5,
        }
        priority += severity_bonus.get(technique.severity, 1.0)

        # Detection difficulty — harder to detect = more interesting to test
        difficulty_bonus = {
            "very_hard": 1.5,
            "hard": 1.0,
            "moderate": 0.5,
            "easy": 0.0,
            "trivial": -0.5,
        }
        priority += difficulty_bonus.get(technique.detection_difficulty, 0.5)

        # Penalty if already tried and failed
        if technique_id in self._failed_techniques:
            priority -= 3.0

        # Penalty if already succeeded (already proven, lower priority)
        if technique_id in self._succeeded_techniques:
            priority -= 1.0

        return max(0.0, priority)

    def _linear_select(
        self, available: list[Probe], max_probes: int
    ) -> list[Probe]:
        """Linear strategy: return probes in their given order.

        Args:
            available: Available probes.
            max_probes: Maximum to return.

        Returns:
            First N probes in order.
        """
        return available[:max_probes]

    def _adaptive_select(
        self, available: list[Probe], max_probes: int
    ) -> list[Probe]:
        """Adaptive strategy: prioritize based on results.

        If a technique succeeded, prioritize probes for techniques that list
        it as a prerequisite. If a technique failed, deprioritize similar ones.

        Args:
            available: Available probes.
            max_probes: Maximum to return.

        Returns:
            Probes sorted by priority score.
        """
        scored = [
            (self._compute_technique_priority(p.technique_id), p)
            for p in available
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:max_probes]]

    def _depth_first_select(
        self, available: list[Probe], max_probes: int
    ) -> list[Probe]:
        """Depth-first strategy: follow successful attack chains deeper.

        When a technique succeeds, prioritize probes for techniques in its
        attack chain (using taxonomy.get_attack_chain).

        Args:
            available: Available probes.
            max_probes: Maximum to return.

        Returns:
            Probes prioritizing chain depth.
        """
        # Build set of "follow-up" technique IDs: techniques whose
        # prerequisites include a succeeded technique
        chain_technique_ids: set[str] = set()
        for succeeded_id in self._succeeded_techniques:
            for technique in self.taxonomy.all_techniques():
                if succeeded_id in technique.prerequisites:
                    chain_technique_ids.add(technique.id)

        # Partition probes: chain-follow probes first
        chain_probes: list[Probe] = []
        other_probes: list[Probe] = []

        for probe in available:
            if probe.technique_id in chain_technique_ids:
                chain_probes.append(probe)
            else:
                chain_probes_scored = self._compute_technique_priority(
                    probe.technique_id
                )
                other_probes.append(probe)

        # Sort chain probes by priority, then append others
        chain_probes.sort(
            key=lambda p: self._compute_technique_priority(p.technique_id),
            reverse=True,
        )
        other_probes.sort(
            key=lambda p: self._compute_technique_priority(p.technique_id),
            reverse=True,
        )

        result = chain_probes + other_probes
        return result[:max_probes]

    def _breadth_first_select(
        self, available: list[Probe], max_probes: int
    ) -> list[Probe]:
        """Breadth-first strategy: ensure all tactic categories are tried.

        Distributes probes across tactic categories before going deeper
        on any single one. Prioritizes untested categories.

        Args:
            available: Available probes.
            max_probes: Maximum to return.

        Returns:
            Probes distributed across tactic categories.
        """
        # Group available probes by tactic
        by_tactic: dict[TacticCategory, list[Probe]] = defaultdict(list)
        uncategorized: list[Probe] = []

        for probe in available:
            try:
                technique = self.taxonomy.get(probe.technique_id)
                by_tactic[technique.tactic].append(probe)
            except KeyError:
                uncategorized.append(probe)

        # Sort tactic categories: untested first, then fewest attempts
        tactic_order = sorted(
            by_tactic.keys(),
            key=lambda t: self._tactic_attempts.get(t, 0),
        )

        # Round-robin across tactics
        result: list[Probe] = []
        idx = 0
        while len(result) < max_probes and tactic_order:
            for tactic in list(tactic_order):
                if len(result) >= max_probes:
                    break
                probes_for_tactic = by_tactic[tactic]
                if idx < len(probes_for_tactic):
                    result.append(probes_for_tactic[idx])
                else:
                    tactic_order.remove(tactic)
            idx += 1

        # Fill remaining from uncategorized
        for probe in uncategorized:
            if len(result) >= max_probes:
                break
            result.append(probe)

        return result
