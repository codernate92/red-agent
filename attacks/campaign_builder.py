"""
Master campaign builder for assembling red-team campaigns from attack suites.

Provides convenience methods for full campaigns, targeted campaigns,
quick scans, and deep dives into specific tactic categories.
"""

from __future__ import annotations

import logging
from typing import Any

from core.campaign import CampaignConfig, CampaignPhase, EscalationStrategy
from core.target import TargetProfile
from core.taxonomy import DEFAULT_TAXONOMY, TacticCategory, TechniqueRegistry

from .defense_evasion import DefenseEvasionSuite
from .exfiltration import ExfiltrationSuite
from .goal_hijacking import GoalHijackingSuite
from .persistence import PersistenceSuite
from .privilege_escalation import PrivilegeEscalationSuite
from .prompt_injection import PromptInjectionSuite

logger = logging.getLogger(__name__)

# Map tactic categories to their suite classes and phase order for full campaigns.
_TACTIC_SUITE_MAP: dict[TacticCategory, type] = {
    TacticCategory.INITIAL_ACCESS: PromptInjectionSuite,
    TacticCategory.PRIVILEGE_ESCALATION: PrivilegeEscalationSuite,
    TacticCategory.DEFENSE_EVASION: DefenseEvasionSuite,
    TacticCategory.PERSISTENCE: PersistenceSuite,
    TacticCategory.EXFILTRATION: ExfiltrationSuite,
    TacticCategory.GOAL_HIJACKING: GoalHijackingSuite,
}

# Canonical ordering of phases in a full campaign.
_FULL_CAMPAIGN_ORDER: list[TacticCategory] = [
    TacticCategory.INITIAL_ACCESS,       # recon + initial access
    TacticCategory.PRIVILEGE_ESCALATION,  # escalation
    TacticCategory.DEFENSE_EVASION,       # evasion
    TacticCategory.GOAL_HIJACKING,        # goal subversion
    TacticCategory.PERSISTENCE,           # persistence
    TacticCategory.EXFILTRATION,          # exfiltration
]


class RedTeamCampaignBuilder:
    """Assembles red-team campaigns from individual attack suites.

    Provides four build modes:

    - ``build_full_campaign`` — all attack phases in canonical order.
    - ``build_targeted_campaign`` — only selected tactic categories.
    - ``build_quick_scan`` — 1-2 probes per tactic for fast overview.
    - ``build_deep_dive`` — all probes for a single tactic.

    Args:
        target_profile: Metadata about the agent under evaluation.
        taxonomy: Technique registry for resolving technique metadata.
    """

    def __init__(
        self,
        target_profile: TargetProfile,
        taxonomy: TechniqueRegistry = DEFAULT_TAXONOMY,
    ) -> None:
        self._target = target_profile
        self._taxonomy = taxonomy
        self._suites: dict[TacticCategory, Any] = {}
        for tactic, suite_cls in _TACTIC_SUITE_MAP.items():
            self._suites[tactic] = suite_cls(taxonomy=self._taxonomy)
        logger.info(
            "RedTeamCampaignBuilder initialized for target %s with %d suites",
            target_profile.name,
            len(self._suites),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_phase_for_tactic(self, tactic: TacticCategory) -> CampaignPhase:
        """Build a full campaign phase for a given tactic category.

        Args:
            tactic: The tactic category to build a phase for.

        Returns:
            A CampaignPhase with all probes from the corresponding suite.

        Raises:
            ValueError: If no suite is registered for the tactic.
        """
        if tactic not in self._suites:
            raise ValueError(
                f"No attack suite registered for tactic {tactic.value!r}. "
                f"Available: {[t.value for t in self._suites]}"
            )
        suite = self._suites[tactic]
        return suite.build_phase(self._target)

    def _build_quick_phase_for_tactic(self, tactic: TacticCategory) -> CampaignPhase:
        """Build a reduced campaign phase with 1-2 probes for fast scanning.

        Args:
            tactic: The tactic category.

        Returns:
            A CampaignPhase with at most 2 probes.
        """
        full_phase = self._build_phase_for_tactic(tactic)
        # Take the first 2 probes as representative samples.
        quick_probes = full_phase.probes[:2]
        return CampaignPhase(
            name=f"quick_{full_phase.name}",
            description=f"Quick scan: {full_phase.description}",
            probes=quick_probes,
            continue_on_failure=True,
            success_threshold=0.0,
        )

    # ------------------------------------------------------------------
    # Public build methods
    # ------------------------------------------------------------------

    def build_full_campaign(self) -> CampaignConfig:
        """Build a full red-team campaign covering all attack categories.

        Phases are ordered: initial access, privilege escalation, defense
        evasion, goal hijacking, persistence, exfiltration.

        Returns:
            A CampaignConfig with all phases and probes.
        """
        phases: list[CampaignPhase] = []
        for tactic in _FULL_CAMPAIGN_ORDER:
            phases.append(self._build_phase_for_tactic(tactic))

        total_probes = sum(len(p.probes) for p in phases)
        config = CampaignConfig(
            name=f"full_red_team_{self._target.name}",
            description=(
                f"Comprehensive red-team evaluation of {self._target.name} across "
                f"{len(phases)} attack phases with {total_probes} total probes."
            ),
            target_name=self._target.name,
            phases=phases,
            escalation_strategy=EscalationStrategy.LINEAR,
            max_total_probes=total_probes + 10,
            timeout_per_probe=60.0,
            record_all_exchanges=True,
        )

        logger.info(
            "Built full campaign %s: %d phases, %d probes",
            config.name,
            len(phases),
            total_probes,
        )
        return config

    def build_targeted_campaign(
        self, tactics: list[TacticCategory]
    ) -> CampaignConfig:
        """Build a campaign targeting only the specified tactic categories.

        Args:
            tactics: List of tactic categories to include.

        Returns:
            A CampaignConfig with phases for the specified tactics only.

        Raises:
            ValueError: If any tactic has no registered suite.
        """
        phases: list[CampaignPhase] = []
        for tactic in tactics:
            phases.append(self._build_phase_for_tactic(tactic))

        total_probes = sum(len(p.probes) for p in phases)
        tactic_names = ", ".join(t.value for t in tactics)
        config = CampaignConfig(
            name=f"targeted_red_team_{self._target.name}",
            description=(
                f"Targeted red-team evaluation of {self._target.name} covering "
                f"tactics: {tactic_names}. {total_probes} total probes."
            ),
            target_name=self._target.name,
            phases=phases,
            escalation_strategy=EscalationStrategy.LINEAR,
            max_total_probes=total_probes + 10,
            timeout_per_probe=60.0,
            record_all_exchanges=True,
        )

        logger.info(
            "Built targeted campaign %s: %d phases (%s), %d probes",
            config.name,
            len(phases),
            tactic_names,
            total_probes,
        )
        return config

    def build_quick_scan(self) -> CampaignConfig:
        """Build a fast overview campaign with 1-2 probes per tactic.

        Useful for initial reconnaissance or CI/CD integration where
        full campaigns are too slow.

        Returns:
            A CampaignConfig with reduced probe count.
        """
        phases: list[CampaignPhase] = []
        for tactic in _FULL_CAMPAIGN_ORDER:
            phases.append(self._build_quick_phase_for_tactic(tactic))

        total_probes = sum(len(p.probes) for p in phases)
        config = CampaignConfig(
            name=f"quick_scan_{self._target.name}",
            description=(
                f"Quick-scan red-team evaluation of {self._target.name}. "
                f"1-2 representative probes per tactic ({total_probes} total)."
            ),
            target_name=self._target.name,
            phases=phases,
            escalation_strategy=EscalationStrategy.BREADTH_FIRST,
            max_total_probes=total_probes + 5,
            timeout_per_probe=30.0,
            cool_down_between_probes=0.2,
            record_all_exchanges=True,
        )

        logger.info(
            "Built quick scan %s: %d phases, %d probes",
            config.name,
            len(phases),
            total_probes,
        )
        return config

    def build_deep_dive(self, tactic: TacticCategory) -> CampaignConfig:
        """Build a deep-dive campaign with all probes for a single tactic.

        Useful when a quick scan reveals a potentially vulnerable area
        that warrants thorough investigation.

        Args:
            tactic: The tactic category to deep-dive into.

        Returns:
            A CampaignConfig with a single phase containing all probes.

        Raises:
            ValueError: If no suite is registered for the tactic.
        """
        phase = self._build_phase_for_tactic(tactic)
        total_probes = len(phase.probes)

        config = CampaignConfig(
            name=f"deep_dive_{tactic.value}_{self._target.name}",
            description=(
                f"Deep-dive red-team evaluation of {self._target.name} focusing "
                f"on {tactic.value}. {total_probes} probes exploring all "
                f"registered techniques."
            ),
            target_name=self._target.name,
            phases=[phase],
            escalation_strategy=EscalationStrategy.DEPTH_FIRST,
            max_total_probes=total_probes + 5,
            timeout_per_probe=90.0,
            record_all_exchanges=True,
        )

        logger.info(
            "Built deep dive %s: 1 phase (%s), %d probes",
            config.name,
            tactic.value,
            total_probes,
        )
        return config
