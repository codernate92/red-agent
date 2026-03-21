"""
Red-team trajectory capture and analysis.

Records a chronological trace of all events during an adversarial campaign:
probes sent, responses received, success evaluations, escalation decisions,
and phase transitions. Trajectories are the primary artifact for post-hoc
analysis and reporting.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .probe import Probe

logger = logging.getLogger(__name__)


@dataclass
class RedTeamEvent:
    """A single event in a red-team trajectory.

    Attributes:
        timestamp: Unix epoch timestamp.
        event_type: Category of event. One of ``"probe_sent"``,
            ``"target_response"``, ``"success_evaluation"``,
            ``"escalation_decision"``, ``"phase_transition"``,
            ``"technique_discovered"``, ``"attack_chain_extended"``.
        probe_id: Associated probe ID, if applicable.
        technique_id: Associated technique ID, if applicable.
        data: Event-specific payload data.
        duration_ms: Duration in milliseconds, if applicable.
    """

    timestamp: float
    event_type: str
    probe_id: str | None = None
    technique_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    _VALID_EVENT_TYPES = frozenset({
        "probe_sent",
        "target_response",
        "success_evaluation",
        "escalation_decision",
        "phase_transition",
        "technique_discovered",
        "attack_chain_extended",
    })

    def __post_init__(self) -> None:
        if self.event_type not in self._VALID_EVENT_TYPES:
            raise ValueError(
                f"event_type must be one of {sorted(self._VALID_EVENT_TYPES)}, "
                f"got {self.event_type!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "probe_id": self.probe_id,
            "technique_id": self.technique_id,
            "data": dict(self.data),
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RedTeamEvent:
        """Deserialize from a plain dictionary."""
        return cls(
            timestamp=data["timestamp"],
            event_type=data["event_type"],
            probe_id=data.get("probe_id"),
            technique_id=data.get("technique_id"),
            data=data.get("data", {}),
            duration_ms=data.get("duration_ms"),
        )


class RedTeamTrajectory:
    """Chronological record of all events during a red-team campaign.

    Provides structured recording methods for each event type and
    analytical queries over the recorded trajectory.

    Args:
        campaign_name: Name of the campaign this trajectory belongs to.
        target_name: Name of the target agent being evaluated.
    """

    def __init__(self, campaign_name: str, target_name: str) -> None:
        self.campaign_name = campaign_name
        self.target_name = target_name
        self.events: list[RedTeamEvent] = []
        self.start_time: float = time.time()
        self.end_time: float | None = None

    def _record(self, event: RedTeamEvent) -> None:
        """Append an event to the trajectory."""
        self.events.append(event)
        logger.debug(
            "Trajectory [%s/%s] recorded %s (probe=%s, technique=%s)",
            self.campaign_name,
            self.target_name,
            event.event_type,
            event.probe_id,
            event.technique_id,
        )

    def record_probe_sent(self, probe: Probe, rendered_payload: str) -> None:
        """Record that a probe was sent to the target agent.

        Args:
            probe: The probe that was sent.
            rendered_payload: The fully rendered payload text.
        """
        self._record(
            RedTeamEvent(
                timestamp=time.time(),
                event_type="probe_sent",
                probe_id=probe.id,
                technique_id=probe.technique_id,
                data={
                    "probe_name": probe.name,
                    "payload": rendered_payload,
                    "setup_turns": probe.setup_turns,
                    "max_turns": probe.max_turns,
                },
            )
        )

    def record_target_response(
        self, probe_id: str, response: str, duration_ms: float
    ) -> None:
        """Record the target agent's response to a probe.

        Args:
            probe_id: ID of the probe that triggered this response.
            response: The agent's response text.
            duration_ms: How long the agent took to respond.
        """
        self._record(
            RedTeamEvent(
                timestamp=time.time(),
                event_type="target_response",
                probe_id=probe_id,
                data={"response": response},
                duration_ms=duration_ms,
            )
        )

    def record_success_evaluation(
        self,
        probe_id: str,
        technique_id: str,
        success: bool,
        confidence: float,
        indicators: list[str],
    ) -> None:
        """Record the success/failure evaluation of a probe.

        Args:
            probe_id: ID of the evaluated probe.
            technique_id: Technique that was tested.
            success: Whether the technique succeeded.
            confidence: Confidence in the evaluation (0.0-1.0).
            indicators: Which success criteria matched.
        """
        self._record(
            RedTeamEvent(
                timestamp=time.time(),
                event_type="success_evaluation",
                probe_id=probe_id,
                technique_id=technique_id,
                data={
                    "success": success,
                    "confidence": confidence,
                    "indicators": indicators,
                },
            )
        )

    def record_escalation_decision(
        self, from_technique: str, to_technique: str, reason: str
    ) -> None:
        """Record a decision to escalate from one technique to another.

        Args:
            from_technique: Technique ID that just succeeded.
            to_technique: Technique ID to attempt next.
            reason: Why this escalation path was chosen.
        """
        self._record(
            RedTeamEvent(
                timestamp=time.time(),
                event_type="escalation_decision",
                technique_id=to_technique,
                data={
                    "from_technique": from_technique,
                    "to_technique": to_technique,
                    "reason": reason,
                },
            )
        )

    def record_phase_transition(
        self, from_phase: str, to_phase: str, results_summary: dict[str, Any]
    ) -> None:
        """Record a transition between campaign phases.

        Args:
            from_phase: Name of the phase being completed.
            to_phase: Name of the phase being entered.
            results_summary: Summary statistics from the completed phase.
        """
        self._record(
            RedTeamEvent(
                timestamp=time.time(),
                event_type="phase_transition",
                data={
                    "from_phase": from_phase,
                    "to_phase": to_phase,
                    "results_summary": results_summary,
                },
            )
        )

    def finalize(self) -> None:
        """Mark the trajectory as complete by recording the end time."""
        self.end_time = time.time()

    def get_successful_probes(self) -> list[RedTeamEvent]:
        """Return all success evaluation events where the technique succeeded.

        Returns:
            List of ``success_evaluation`` events with ``success=True``.
        """
        return [
            e
            for e in self.events
            if e.event_type == "success_evaluation" and e.data.get("success") is True
        ]

    def get_attack_timeline(self) -> list[dict[str, Any]]:
        """Build a chronological view of the attack as a list of dicts.

        Returns:
            Ordered list of event summaries with timestamp, type, and key data.
        """
        timeline: list[dict[str, Any]] = []
        for event in sorted(self.events, key=lambda e: e.timestamp):
            entry: dict[str, Any] = {
                "timestamp": event.timestamp,
                "event_type": event.event_type,
            }
            if event.probe_id:
                entry["probe_id"] = event.probe_id
            if event.technique_id:
                entry["technique_id"] = event.technique_id
            if event.duration_ms is not None:
                entry["duration_ms"] = event.duration_ms

            # Include salient data fields depending on event type.
            if event.event_type == "probe_sent":
                entry["payload_preview"] = event.data.get("payload", "")[:200]
            elif event.event_type == "target_response":
                entry["response_preview"] = event.data.get("response", "")[:200]
            elif event.event_type == "success_evaluation":
                entry["success"] = event.data.get("success")
                entry["confidence"] = event.data.get("confidence")
            elif event.event_type == "escalation_decision":
                entry["from_technique"] = event.data.get("from_technique")
                entry["to_technique"] = event.data.get("to_technique")
            elif event.event_type == "phase_transition":
                entry["from_phase"] = event.data.get("from_phase")
                entry["to_phase"] = event.data.get("to_phase")

            timeline.append(entry)
        return timeline

    def get_technique_success_rates(self) -> dict[str, float]:
        """Compute success rate per technique across all probes.

        Returns:
            Map of technique ID to success rate (0.0-1.0).
        """
        technique_counts: dict[str, int] = {}
        technique_successes: dict[str, int] = {}

        for event in self.events:
            if event.event_type == "success_evaluation" and event.technique_id:
                tid = event.technique_id
                technique_counts[tid] = technique_counts.get(tid, 0) + 1
                if event.data.get("success"):
                    technique_successes[tid] = technique_successes.get(tid, 0) + 1

        return {
            tid: technique_successes.get(tid, 0) / count
            for tid, count in technique_counts.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full trajectory to a dictionary."""
        return {
            "campaign_name": self.campaign_name,
            "target_name": self.target_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RedTeamTrajectory:
        """Deserialize from a dictionary."""
        traj = cls(
            campaign_name=data["campaign_name"],
            target_name=data["target_name"],
        )
        traj.start_time = data["start_time"]
        traj.end_time = data.get("end_time")
        traj.events = [RedTeamEvent.from_dict(e) for e in data.get("events", [])]
        return traj

    def to_json(self, indent: int = 2) -> str:
        """Serialize the full trajectory to a JSON string.

        Args:
            indent: JSON indentation level.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> RedTeamTrajectory:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(json_str))


class RedTeamTrajectoryStore:
    """In-memory store for multiple red-team trajectories.

    Trajectories are keyed by ``(campaign_name, target_name)`` pairs.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], RedTeamTrajectory] = {}

    def store(self, trajectory: RedTeamTrajectory) -> None:
        """Store a trajectory, keyed by campaign and target name.

        Args:
            trajectory: The trajectory to store.
        """
        key = (trajectory.campaign_name, trajectory.target_name)
        self._store[key] = trajectory
        logger.debug(
            "Stored trajectory for campaign=%s target=%s (%d events)",
            trajectory.campaign_name,
            trajectory.target_name,
            len(trajectory.events),
        )

    def get(
        self, campaign_name: str, target_name: str
    ) -> RedTeamTrajectory | None:
        """Retrieve a trajectory by campaign and target name.

        Args:
            campaign_name: Campaign identifier.
            target_name: Target agent identifier.

        Returns:
            The trajectory, or ``None`` if not found.
        """
        return self._store.get((campaign_name, target_name))

    def list_keys(self) -> list[tuple[str, str]]:
        """List all stored (campaign_name, target_name) keys.

        Returns:
            List of key tuples.
        """
        return list(self._store.keys())

    def get_by_campaign(self, campaign_name: str) -> list[RedTeamTrajectory]:
        """Retrieve all trajectories for a given campaign.

        Args:
            campaign_name: Campaign identifier.

        Returns:
            List of trajectories matching the campaign name.
        """
        return [
            traj
            for (cname, _), traj in self._store.items()
            if cname == campaign_name
        ]

    def get_by_target(self, target_name: str) -> list[RedTeamTrajectory]:
        """Retrieve all trajectories for a given target.

        Args:
            target_name: Target agent identifier.

        Returns:
            List of trajectories matching the target name.
        """
        return [
            traj
            for (_, tname), traj in self._store.items()
            if tname == target_name
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize all stored trajectories."""
        return {
            "trajectories": [
                traj.to_dict() for traj in self._store.values()
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RedTeamTrajectoryStore:
        """Deserialize from a dictionary."""
        store = cls()
        for traj_data in data.get("trajectories", []):
            traj = RedTeamTrajectory.from_dict(traj_data)
            store.store(traj)
        return store

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._store
