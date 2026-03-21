"""
Full red-team report generation for LLM agent evaluations.

Produces professional-grade reports with executive summaries, vulnerability
details, attack surface analysis, attack timelines, and technique matrices.
Supports JSON and Markdown output formats.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from core.campaign import CampaignResult
from core.probe import ProbeStatus
from core.taxonomy import TacticCategory, TechniqueRegistry
from core.trajectory import RedTeamTrajectory

from .attack_surface import AttackSurfaceMap
from .vulnerability import VulnerabilityScorer

logger = logging.getLogger(__name__)


class RedTeamReport:
    """Full red-team report generator.

    Combines vulnerability scoring, attack surface mapping, and trajectory
    analysis into a comprehensive report suitable for security teams.

    Args:
        campaign_result: Completed campaign results.
        trajectory: The campaign's trajectory data.
        taxonomy: Technique registry for metadata lookups.
    """

    def __init__(
        self,
        campaign_result: CampaignResult,
        trajectory: RedTeamTrajectory,
        taxonomy: TechniqueRegistry,
    ) -> None:
        self.campaign_result = campaign_result
        self.trajectory = trajectory
        self.taxonomy = taxonomy
        self._scorer = VulnerabilityScorer(campaign_result, taxonomy)
        self._surface = AttackSurfaceMap(campaign_result, taxonomy)

    def executive_summary(self) -> dict[str, Any]:
        """Generate a concise executive summary.

        Returns:
            Dict with key findings, overall risk score, and top vulnerabilities.
        """
        posture = self._scorer.overall_security_posture()
        vulns = self._scorer.score_all()
        resistance = self._surface.resistance_profile()
        chains = self._surface.find_attack_chains()

        findings: list[str] = []

        # Finding 1: Overall risk assessment
        risk = self.campaign_result.overall_risk_score
        if risk >= 9.0:
            findings.append(
                f"CRITICAL: Agent has a risk score of {risk:.1f}/10.0 "
                f"indicating severe security vulnerabilities."
            )
        elif risk >= 7.0:
            findings.append(
                f"HIGH RISK: Agent has a risk score of {risk:.1f}/10.0 "
                f"with significant vulnerabilities requiring immediate attention."
            )
        elif risk >= 4.0:
            findings.append(
                f"MODERATE RISK: Agent has a risk score of {risk:.1f}/10.0 "
                f"with vulnerabilities that should be addressed."
            )
        else:
            findings.append(
                f"LOW RISK: Agent has a risk score of {risk:.1f}/10.0 "
                f"with minor or no vulnerabilities detected."
            )

        # Finding 2: Vulnerability count
        total_vulns = posture["total_vulnerabilities"]
        if total_vulns > 0:
            findings.append(
                f"Discovered {total_vulns} vulnerability/vulnerabilities across "
                f"{posture['total_techniques_tested']} tested techniques "
                f"({self.campaign_result.success_rate:.0%} exploitation rate)."
            )

        # Finding 3: Attack chains
        if chains:
            longest = max(chains, key=len)
            chain_names = []
            for tid in longest:
                try:
                    chain_names.append(self.taxonomy.get(tid).name)
                except KeyError:
                    chain_names.append(tid)
            findings.append(
                f"Viable {len(longest)}-step attack chain identified: "
                + " -> ".join(chain_names) + "."
            )

        # Finding 4: Weak areas
        if resistance["weak_resistance"]:
            findings.append(
                f"Weak resistance in: {', '.join(resistance['weak_resistance'])}. "
                f"These areas need priority hardening."
            )

        # Finding 5: Strong areas
        if resistance["strong_resistance"]:
            findings.append(
                f"Strong resistance demonstrated in: "
                f"{', '.join(resistance['strong_resistance'])}."
            )

        top_vulns = [v.to_dict() for v in vulns[:3]]

        return {
            "findings": findings,
            "overall_risk_score": self.campaign_result.overall_risk_score,
            "top_vulnerabilities": top_vulns,
            "security_posture": posture,
        }

    def vulnerability_details(self) -> list[dict[str, Any]]:
        """Generate detailed vulnerability information.

        Returns:
            List of dicts, one per vulnerability, with full details.
        """
        vulns = self._scorer.score_all()
        return [v.to_dict() for v in vulns]

    def attack_surface_analysis(self) -> dict[str, Any]:
        """Generate attack surface analysis.

        Returns:
            Dict with surface map and coverage gaps.
        """
        return self._surface.to_dict()

    def attack_timeline(self) -> list[dict[str, Any]]:
        """Generate a chronological attack narrative.

        Returns:
            Ordered list of event dicts from the trajectory.
        """
        return self.trajectory.get_attack_timeline()

    def technique_matrix(self) -> dict[str, Any]:
        """Generate an ATT&CK-style matrix showing tested/succeeded/failed per tactic.

        Returns:
            Dict mapping tactic categories to their technique outcomes.
        """
        # Collect all results by technique
        technique_outcomes: dict[str, str] = {}  # technique_id -> "succeeded"/"failed"/"partial"/"error"
        for results in self.campaign_result.phase_results.values():
            for result in results:
                tid = result.technique_id
                if result.status == ProbeStatus.SUCCESS:
                    technique_outcomes[tid] = "succeeded"
                elif tid not in technique_outcomes:
                    if result.status == ProbeStatus.PARTIAL:
                        technique_outcomes[tid] = "partial"
                    elif result.status == ProbeStatus.FAILED:
                        technique_outcomes[tid] = "failed"
                    elif result.status == ProbeStatus.ERROR:
                        technique_outcomes[tid] = "error"

        matrix: dict[str, Any] = {}
        for tactic in TacticCategory:
            techniques_in_tactic = self.taxonomy.get_by_tactic(tactic)
            entries: list[dict[str, Any]] = []

            for tech in techniques_in_tactic:
                outcome = technique_outcomes.get(tech.id, "not_tested")
                entries.append({
                    "technique_id": tech.id,
                    "technique_name": tech.name,
                    "severity": tech.severity,
                    "outcome": outcome,
                })

            tested = sum(1 for e in entries if e["outcome"] != "not_tested")
            succeeded = sum(1 for e in entries if e["outcome"] == "succeeded")

            matrix[tactic.value] = {
                "tactic_name": tactic.name,
                "techniques": entries,
                "total": len(entries),
                "tested": tested,
                "succeeded": succeeded,
            }

        return matrix

    def full_report(self) -> dict[str, Any]:
        """Generate the complete report with all sections.

        Returns:
            Dict with all report sections.
        """
        return {
            "report_metadata": {
                "campaign_name": self.campaign_result.campaign_name,
                "target_name": self.campaign_result.target_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": self.campaign_result.duration_seconds,
                "total_probes": self.campaign_result.total_probes,
            },
            "executive_summary": self.executive_summary(),
            "vulnerability_details": self.vulnerability_details(),
            "attack_surface_analysis": self.attack_surface_analysis(),
            "attack_timeline": self.attack_timeline(),
            "technique_matrix": self.technique_matrix(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the full report to JSON.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string of the full report.
        """
        return json.dumps(self.full_report(), indent=indent)

    def to_markdown(self) -> str:
        """Generate a professional Markdown red-team report.

        Returns:
            Markdown string suitable for delivery to a security team.
        """
        report = self.full_report()
        meta = report["report_metadata"]
        exec_summary = report["executive_summary"]
        vulns = report["vulnerability_details"]
        surface = report["attack_surface_analysis"]
        matrix = report["technique_matrix"]

        lines: list[str] = []

        # Title
        lines.append("# Red-Team Assessment Report")
        lines.append("")
        lines.append(f"**Target:** {meta['target_name']}  ")
        lines.append(f"**Campaign:** {meta['campaign_name']}  ")
        lines.append(f"**Date:** {meta['generated_at'][:10]}  ")
        lines.append(f"**Duration:** {meta['duration_seconds']:.1f} seconds  ")
        lines.append(f"**Total Probes:** {meta['total_probes']}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        risk_score = exec_summary["overall_risk_score"]
        lines.append(f"**Overall Risk Score: {risk_score:.1f} / 10.0**")
        lines.append("")

        posture = exec_summary.get("security_posture", {})
        if posture:
            lines.append(f"- **Vulnerabilities Found:** {posture.get('total_vulnerabilities', 0)}")
            lines.append(f"- **Techniques Tested:** {posture.get('total_techniques_tested', 0)}")
            lines.append(f"- **Exploitation Rate:** {posture.get('success_rate', 0):.0%}")
            lines.append("")

        lines.append("### Key Findings")
        lines.append("")
        for i, finding in enumerate(exec_summary.get("findings", []), 1):
            lines.append(f"{i}. {finding}")
        lines.append("")

        # Findings Summary Table
        if vulns:
            lines.append("---")
            lines.append("")
            lines.append("## Vulnerability Summary")
            lines.append("")
            lines.append("| # | Technique | Tactic | Severity | CVSS Score | Exploitability |")
            lines.append("|---|-----------|--------|----------|------------|----------------|")
            for i, vuln in enumerate(vulns, 1):
                severity_str = vuln["severity"].upper()
                lines.append(
                    f"| {i} | {vuln['technique_name']} | {vuln['tactic']} | "
                    f"{severity_str} | {vuln['cvss_analog_score']:.1f} | "
                    f"{vuln['exploitability']:.0%} |"
                )
            lines.append("")

        # Detailed Vulnerabilities
        if vulns:
            lines.append("---")
            lines.append("")
            lines.append("## Detailed Vulnerability Analysis")
            lines.append("")
            for i, vuln in enumerate(vulns, 1):
                severity_str = vuln["severity"].upper()
                lines.append(f"### {i}. {vuln['technique_name']} ({vuln['technique_id']})")
                lines.append("")
                lines.append(f"- **Severity:** {severity_str}")
                lines.append(f"- **CVSS Analog Score:** {vuln['cvss_analog_score']:.1f}")
                lines.append(f"- **Tactic:** {vuln['tactic']}")
                lines.append(f"- **Exploitability:** {vuln['exploitability']:.0%}")
                lines.append(f"- **Impact:** {vuln['impact']}")
                lines.append("")
                lines.append(f"**Remediation:** {vuln['remediation']}")
                lines.append("")
                if vuln.get("proof_probes"):
                    lines.append(f"*Demonstrated by {len(vuln['proof_probes'])} probe(s).*")
                    lines.append("")

        # Technique Matrix
        lines.append("---")
        lines.append("")
        lines.append("## Technique Matrix")
        lines.append("")
        lines.append(
            "ATT&CK-style matrix showing the outcome of each technique "
            "tested during the campaign."
        )
        lines.append("")

        for tactic_name, tactic_data in matrix.items():
            tactic_display = tactic_name.replace("_", " ").title()
            tested = tactic_data["tested"]
            succeeded = tactic_data["succeeded"]
            total = tactic_data["total"]

            lines.append(f"### {tactic_display} ({tested}/{total} tested, {succeeded} succeeded)")
            lines.append("")

            if tactic_data["techniques"]:
                lines.append("| Technique | Severity | Outcome |")
                lines.append("|-----------|----------|---------|")
                for tech in tactic_data["techniques"]:
                    outcome = tech["outcome"]
                    outcome_display = {
                        "succeeded": "EXPLOITED",
                        "failed": "RESISTED",
                        "partial": "PARTIAL",
                        "error": "ERROR",
                        "not_tested": "-",
                    }.get(outcome, outcome)
                    lines.append(
                        f"| {tech['technique_name']} ({tech['technique_id']}) | "
                        f"{tech['severity'].upper()} | {outcome_display} |"
                    )
                lines.append("")

        # Attack Surface Analysis
        lines.append("---")
        lines.append("")
        lines.append("## Attack Surface Analysis")
        lines.append("")

        surface_map = surface.get("surface_map", {})
        lines.append("| Tactic Category | Tested | Vulnerable | Techniques Tested | Success Rate |")
        lines.append("|-----------------|--------|------------|-------------------|--------------|")
        for tactic_name, info in surface_map.items():
            tactic_display = tactic_name.replace("_", " ").title()
            tested_str = "Yes" if info["tested"] else "No"
            vuln_str = "YES" if info["vulnerable"] else "No"
            lines.append(
                f"| {tactic_display} | {tested_str} | {vuln_str} | "
                f"{info['techniques_tested']} | {info['success_rate']:.0%} |"
            )
        lines.append("")

        # Coverage Gaps
        gaps = surface.get("coverage_gaps", [])
        if gaps:
            lines.append("### Coverage Gaps")
            lines.append("")
            lines.append(
                "The following tactic categories were not tested or under-tested:"
            )
            lines.append("")
            for gap in gaps:
                lines.append(f"- {gap.replace('_', ' ').title()}")
            lines.append("")

        # Attack Chains
        chains = surface.get("attack_chains", [])
        if chains:
            lines.append("### Attack Chains")
            lines.append("")
            lines.append(
                "The following multi-step attack paths were successfully demonstrated:"
            )
            lines.append("")
            for i, chain in enumerate(chains, 1):
                chain_names = []
                for tid in chain:
                    try:
                        tech = self.taxonomy.get(tid)
                        chain_names.append(f"{tech.name} ({tid})")
                    except KeyError:
                        chain_names.append(tid)
                lines.append(f"{i}. " + " --> ".join(chain_names))
            lines.append("")

        # Resistance Profile
        resistance = surface.get("resistance_profile", {})
        if resistance:
            lines.append("### Resistance Profile")
            lines.append("")
            score = resistance.get("overall_resistance_score", 0)
            lines.append(f"**Overall Resistance Score:** {score:.0%}")
            lines.append("")

            if resistance.get("strong_resistance"):
                lines.append(
                    "**Strong Resistance:** "
                    + ", ".join(r.replace("_", " ").title() for r in resistance["strong_resistance"])
                )
            if resistance.get("weak_resistance"):
                lines.append(
                    "**Weak Resistance:** "
                    + ", ".join(r.replace("_", " ").title() for r in resistance["weak_resistance"])
                )
            if resistance.get("partial_resistance"):
                lines.append(
                    "**Partial Resistance:** "
                    + ", ".join(r.replace("_", " ").title() for r in resistance["partial_resistance"])
                )
            lines.append("")

        # Recommendations
        lines.append("---")
        lines.append("")
        lines.append("## Recommendations")
        lines.append("")

        if vulns:
            lines.append("### Priority Remediation Actions")
            lines.append("")
            for i, vuln in enumerate(vulns[:5], 1):
                lines.append(
                    f"{i}. **{vuln['technique_name']}** ({vuln['severity'].upper()}): "
                    f"{vuln['remediation']}"
                )
            lines.append("")

        lines.append("### General Recommendations")
        lines.append("")
        lines.append("1. Implement defense-in-depth with multiple safety layers.")
        lines.append("2. Conduct regular red-team assessments with updated attack techniques.")
        lines.append("3. Monitor agent outputs for anomalous behavior patterns.")
        lines.append("4. Establish an incident response plan for detected exploitation attempts.")
        lines.append("5. Keep safety training data and alignment techniques current.")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(
            f"*Report generated by red-agent adversarial assessment framework.*"
        )

        return "\n".join(lines)
