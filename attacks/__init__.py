"""
Attack library for the red-agent adversarial evaluation framework.

Re-exports all attack suite classes and the master campaign builder for
convenient top-level access::

    from attacks import PromptInjectionSuite, RedTeamCampaignBuilder
"""

from .campaign_builder import RedTeamCampaignBuilder
from .defense_evasion import DefenseEvasionSuite
from .exfiltration import ExfiltrationSuite
from .goal_hijacking import GoalHijackingSuite
from .persistence import PersistenceSuite
from .privilege_escalation import PrivilegeEscalationSuite
from .prompt_injection import PromptInjectionSuite

__all__ = [
    "PromptInjectionSuite",
    "PrivilegeEscalationSuite",
    "GoalHijackingSuite",
    "ExfiltrationSuite",
    "DefenseEvasionSuite",
    "PersistenceSuite",
    "RedTeamCampaignBuilder",
]
