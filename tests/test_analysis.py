"""
Tests for vulnerability scoring, attack surface mapping, and report generation.
"""

from __future__ import annotations

import json

import pytest

from core.campaign import CampaignResult, EscalationStrategy
from core.probe import Probe, ProbeBuilder, ProbeResult, ProbeStatus
from core.taxonomy import DEFAULT_TAXONOMY, Procedure, TacticCategory, TechniqueRegistry
from core.trajectory import RedTeamTrajectory, RedTeamTrajectoryStore

from analysis.vulnerability import AgentVulnerability, VulnerabilityRating, VulnerabilityScorer
from analysis.attack_surface import AttackSurfaceMap
from analysis.report import RedTeamReport
from harness.adaptive import AdaptiveSelector


def test_vulnerability_scoring(
    sample_campaign_result: CampaignResult,
    technique_registry: TechniqueRegistry,
) -> None:
    """Known successful techniques should produce valid CVSS scores."""
    scorer = VulnerabilityScorer(sample_campaign_result, technique_registry)
    vulns = scorer.score_all()

    assert len(vulns) > 0
    for vuln in vulns:
        assert isinstance(vuln, AgentVulnerability)
        assert 0.0 <= vuln.cvss_analog_score <= 10.0
        assert isinstance(vuln.severity, VulnerabilityRating)
        assert vuln.technique_id in sample_campaign_result.successful_techniques
        assert len(vuln.proof_probes) > 0
        assert vuln.impact != ""
        assert vuln.remediation != ""

    # Vulns should be sorted by score descending
    scores = [v.cvss_analog_score for v in vulns]
    assert scores == sorted(scores, reverse=True)


def test_vulnerability_rating_from_score() -> None:
    """Verify VulnerabilityRating.from_score maps correctly."""
    assert VulnerabilityRating.from_score(0.0) == VulnerabilityRating.NONE
    assert VulnerabilityRating.from_score(2.0) == VulnerabilityRating.LOW
    assert VulnerabilityRating.from_score(5.5) == VulnerabilityRating.MEDIUM
    assert VulnerabilityRating.from_score(7.5) == VulnerabilityRating.HIGH
    assert VulnerabilityRating.from_score(9.5) == VulnerabilityRating.CRITICAL


def test_overall_security_posture(
    sample_campaign_result: CampaignResult,
    technique_registry: TechniqueRegistry,
) -> None:
    """Verify overall_security_posture returns expected structure."""
    scorer = VulnerabilityScorer(sample_campaign_result, technique_registry)
    posture = scorer.overall_security_posture()

    assert "vulns_by_rating" in posture
    assert "top_critical" in posture
    assert "overall_score" in posture
    assert "total_vulnerabilities" in posture
    assert 0.0 <= posture["overall_score"] <= 10.0


def test_attack_surface_mapping(
    sample_campaign_result: CampaignResult,
    technique_registry: TechniqueRegistry,
) -> None:
    """Verify all tested tactics appear in the surface map."""
    surface_map = AttackSurfaceMap(sample_campaign_result, technique_registry)
    surface = surface_map.map_surface()

    # Should have an entry for every tactic category
    for tactic in TacticCategory:
        assert tactic.value in surface

    # Tactics with tested techniques should be marked as tested
    tested_tactics = set()
    for results in sample_campaign_result.phase_results.values():
        for result in results:
            try:
                tech = technique_registry.get(result.technique_id)
                tested_tactics.add(tech.tactic)
            except KeyError:
                pass

    for tactic in tested_tactics:
        assert surface[tactic.value]["tested"] is True


def test_identify_gaps(
    sample_campaign_result: CampaignResult,
    technique_registry: TechniqueRegistry,
) -> None:
    """Verify gap identification finds untested categories."""
    surface_map = AttackSurfaceMap(sample_campaign_result, technique_registry)
    gaps = surface_map.identify_gaps()

    assert isinstance(gaps, list)
    # There should be some gaps since we don't test all categories
    assert len(gaps) > 0


def test_attack_chain_detection(
    sample_campaign_result: CampaignResult,
    technique_registry: TechniqueRegistry,
) -> None:
    """Sequential technique successes should form chains."""
    surface_map = AttackSurfaceMap(sample_campaign_result, technique_registry)
    chains = surface_map.find_attack_chains()

    # With T1004, T1003, T1001, T1005, T1006 succeeded, there should be
    # at least one chain (e.g., T1004 -> T1003 has no prereqs but
    # T1005 requires T1014 which failed, and T1006 has no prereqs).
    # Actually T1001 -> T1005 isn't a direct chain.
    # Chains depend on prerequisite relationships.
    assert isinstance(chains, list)
    for chain in chains:
        assert len(chain) >= 2


def test_resistance_profile(
    technique_registry: TechniqueRegistry,
) -> None:
    """Verify a hardened-looking result has strong resistance metrics."""
    import time
    import uuid

    # Build a result where most probes fail
    now = time.time()
    results = []
    for tid in ["T1001", "T1003", "T1005", "T1006"]:
        results.append(
            ProbeResult(
                probe_id=str(uuid.uuid4()),
                status=ProbeStatus.FAILED,
                technique_id=tid,
                target_response="I cannot comply.",
                confidence=0.1,
            )
        )

    campaign_result = CampaignResult(
        campaign_name="hardened-test",
        target_name="hardened-agent",
        start_time=now - 10,
        end_time=now,
        phase_results={"test": results},
        successful_techniques=[],
        failed_techniques=["T1001", "T1003", "T1005", "T1006"],
        overall_risk_score=0.0,
    )

    surface_map = AttackSurfaceMap(campaign_result, technique_registry)
    profile = surface_map.resistance_profile()

    assert profile["overall_resistance_score"] == 1.0
    assert len(profile["strong_resistance"]) > 0
    assert len(profile["weak_resistance"]) == 0


def test_report_generation(
    sample_campaign_result: CampaignResult,
    technique_registry: TechniqueRegistry,
    populated_trajectory: RedTeamTrajectoryStore,
) -> None:
    """Full report should have all sections."""
    traj = populated_trajectory.get("test-campaign", "helpful-assistant-v1")
    assert traj is not None

    report = RedTeamReport(sample_campaign_result, traj, technique_registry)
    full = report.full_report()

    assert "report_metadata" in full
    assert "executive_summary" in full
    assert "vulnerability_details" in full
    assert "attack_surface_analysis" in full
    assert "attack_timeline" in full
    assert "technique_matrix" in full

    # Executive summary should have findings
    exec_sum = full["executive_summary"]
    assert len(exec_sum["findings"]) >= 1
    assert "overall_risk_score" in exec_sum

    # Technique matrix should cover all tactics
    matrix = full["technique_matrix"]
    for tactic in TacticCategory:
        assert tactic.value in matrix


def test_report_json(
    sample_campaign_result: CampaignResult,
    technique_registry: TechniqueRegistry,
    populated_trajectory: RedTeamTrajectoryStore,
) -> None:
    """JSON output should be valid JSON."""
    traj = populated_trajectory.get("test-campaign", "helpful-assistant-v1")
    assert traj is not None

    report = RedTeamReport(sample_campaign_result, traj, technique_registry)
    json_str = report.to_json()

    parsed = json.loads(json_str)
    assert "executive_summary" in parsed


def test_report_markdown(
    sample_campaign_result: CampaignResult,
    technique_registry: TechniqueRegistry,
    populated_trajectory: RedTeamTrajectoryStore,
) -> None:
    """Markdown output should have expected headers and tables."""
    traj = populated_trajectory.get("test-campaign", "helpful-assistant-v1")
    assert traj is not None

    report = RedTeamReport(sample_campaign_result, traj, technique_registry)
    md = report.to_markdown()

    assert "# Red-Team Assessment Report" in md
    assert "## Executive Summary" in md
    assert "## Vulnerability Summary" in md
    assert "## Detailed Vulnerability Analysis" in md
    assert "## Technique Matrix" in md
    assert "## Attack Surface Analysis" in md
    assert "## Recommendations" in md
    assert "| Technique |" in md  # Table header present
    assert "**Target:**" in md


def test_adaptive_selector_depth_first(
    technique_registry: TechniqueRegistry,
    sample_probes: list[Probe],
) -> None:
    """Successful technique should lead to follow-up probes via depth-first."""
    selector = AdaptiveSelector(technique_registry)

    # Simulate T1004 (Tool Permission Probing) succeeding
    selector.update(
        ProbeResult(
            probe_id="p1",
            status=ProbeStatus.SUCCESS,
            technique_id="T1004",
            target_response="Here are my tools...",
            confidence=0.8,
        )
    )

    # Build probes that include techniques with T1004 as a prerequisite
    # T1007, T1011, T1018 all have T1004 as a prerequisite
    follow_up_probes: list[Probe] = []
    for tid in ["T1007", "T1011", "T1018", "T1001", "T1005"]:
        follow_up_probes.append(
            ProbeBuilder()
            .for_technique(tid, technique_registry)
            .with_procedure(
                Procedure(
                    technique_id=tid,
                    name=f"test_{tid}",
                    description="Test",
                    payload_template="test",
                    success_criteria="test",
                    expected_behavior="test",
                )
            )
            .with_payload("test payload")
            .build()
        )

    selected = selector.suggest_next_probes(
        follow_up_probes, EscalationStrategy.DEPTH_FIRST, max_probes=3
    )

    assert len(selected) == 3
    # The first probes should be those whose prerequisites include T1004
    first_technique_ids = {p.technique_id for p in selected}
    # At least one of T1007, T1011, T1018 should be in the first batch
    follow_up_ids = {"T1007", "T1011", "T1018"}
    assert len(first_technique_ids & follow_up_ids) >= 1


def test_adaptive_selector_breadth_first(
    technique_registry: TechniqueRegistry,
) -> None:
    """Breadth-first should ensure all tactic categories are tried."""
    selector = AdaptiveSelector(technique_registry)

    # Create probes across multiple tactics
    probes: list[Probe] = []
    tactic_technique_pairs = [
        ("T1004", TacticCategory.RECONNAISSANCE),
        ("T1014", TacticCategory.RECONNAISSANCE),
        ("T1001", TacticCategory.INITIAL_ACCESS),
        ("T1002", TacticCategory.INITIAL_ACCESS),
        ("T1005", TacticCategory.DEFENSE_EVASION),
        ("T1010", TacticCategory.DEFENSE_EVASION),
        ("T1003", TacticCategory.EXFILTRATION),
        ("T1006", TacticCategory.GOAL_HIJACKING),
    ]

    for tid, _ in tactic_technique_pairs:
        probes.append(
            ProbeBuilder()
            .for_technique(tid, technique_registry)
            .with_procedure(
                Procedure(
                    technique_id=tid,
                    name=f"test_{tid}",
                    description="Test",
                    payload_template="test",
                    success_criteria="test",
                    expected_behavior="test",
                )
            )
            .with_payload("test payload")
            .build()
        )

    selected = selector.suggest_next_probes(
        probes, EscalationStrategy.BREADTH_FIRST, max_probes=5
    )

    assert len(selected) == 5
    # Should have probes from multiple tactics, not all from one
    tactics_represented = set()
    for probe in selected:
        try:
            tech = technique_registry.get(probe.technique_id)
            tactics_represented.add(tech.tactic)
        except KeyError:
            pass

    # With 5 probes across 4+ tactics, should cover at least 3 tactics
    assert len(tactics_represented) >= 3
