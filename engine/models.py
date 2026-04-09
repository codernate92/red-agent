"""Typed models for the vulnerability engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_serializer


class Vulnerability(BaseModel):
    """A confirmed vulnerability discovered by the engine."""

    id: str
    severity: float
    attack_type: str
    technique: str
    reproduction_rate: float
    strongreject_score: float
    example_prompt: str
    example_response: str
    prefill_used: str | None = None
    confidence_interval: tuple[float, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VulnReport(BaseModel):
    """Structured output from a single-model scan."""

    model: str
    scan_duration_seconds: float
    total_probes_sent: int
    vulnerabilities: list[Vulnerability]
    attack_surface_heatmap: dict[tuple[str, str], float]
    recommendations: list[str]

    @field_serializer("attack_surface_heatmap", when_used="json")
    def serialize_heatmap(
        self,
        heatmap: dict[tuple[str, str], float],
    ) -> dict[str, float]:
        return {f"{attack_type}::{topic}": score for (attack_type, topic), score in heatmap.items()}


class CrossModelVulnerability(BaseModel):
    """A vulnerability signature observed across models."""

    signature: str
    models: list[str]
    mean_severity: float
    max_severity: float


class ScalingAnalysis(BaseModel):
    """Simple regression output over known parameter counts."""

    metric: str
    sample_size: int
    slope: float
    intercept: float
    r_squared: float


class ModelVulnerability(BaseModel):
    """A vulnerability annotated with the model where it appeared."""

    model: str
    vulnerability: Vulnerability


class BatchReport(BaseModel):
    """Structured output from scanning multiple models."""

    reports: dict[str, VulnReport]
    cross_model_comparison: dict[str, list[CrossModelVulnerability]]
    scaling_analysis: ScalingAnalysis | None = None
    worst_findings: list[ModelVulnerability]
    errors: dict[str, str] = Field(default_factory=dict)


class ActionItem(BaseModel):
    """An action derived from a vulnerability finding."""

    priority: str
    action: str
    template: str | None = None
    provider: str | None = None
    disclosure_summary: str | None = None
    recommended_fix: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).date().isoformat()
    )
