"""
red-agent core — Adversarial agent red-teaming framework.

Re-exports all key types for convenient access::

    from core import (
        TacticCategory, Technique, Procedure, TechniqueRegistry,
        DEFAULT_TAXONOMY, ProbeStatus, ProbeResult, Probe, ProbeBuilder,
        CampaignPhase, EscalationStrategy, CampaignConfig, CampaignResult,
        TargetAgent, TargetProfile, MockTarget,
        RedTeamEvent, RedTeamTrajectory, RedTeamTrajectoryStore,
    )
"""

from .taxonomy import (
    DEFAULT_TAXONOMY,
    Procedure,
    TacticCategory,
    Technique,
    TechniqueRegistry,
)
from .probe import (
    Probe,
    ProbeBuilder,
    ProbeResult,
    ProbeStatus,
    SuccessDetector,
)
from .campaign import (
    CampaignConfig,
    CampaignPhase,
    CampaignResult,
    EscalationStrategy,
)
from .target import (
    MockTarget,
    TargetAgent,
    TargetProfile,
)
from .trajectory import (
    RedTeamEvent,
    RedTeamTrajectory,
    RedTeamTrajectoryStore,
)

__all__ = [
    # taxonomy
    "TacticCategory",
    "Technique",
    "Procedure",
    "TechniqueRegistry",
    "DEFAULT_TAXONOMY",
    # probe
    "ProbeStatus",
    "ProbeResult",
    "Probe",
    "ProbeBuilder",
    "SuccessDetector",
    # campaign
    "CampaignPhase",
    "EscalationStrategy",
    "CampaignConfig",
    "CampaignResult",
    # target
    "TargetAgent",
    "TargetProfile",
    "MockTarget",
    # trajectory
    "RedTeamEvent",
    "RedTeamTrajectory",
    "RedTeamTrajectoryStore",
]
