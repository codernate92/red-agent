"""Multi-model comparison runner.

Runs an identical campaign across a list of registered models and aggregates
results into a single tabular view suitable for cross-provider comparison and
scaling regressions (learned overall score vs. log parameter count).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

from core.campaign import CampaignConfig, CampaignResult
from core.model_registry import MODEL_REGISTRY, ModelSpec, resolve
from core.probe import ProbeStatus
from core.targets import BaseTarget, TargetError, create_target
from core.taxonomy import DEFAULT_TAXONOMY, TechniqueRegistry
from core.trajectory import RedTeamTrajectoryStore

from .runner import CampaignRunner

logger = logging.getLogger(__name__)


@dataclass
class ModelComparisonResult:
    """Results of a single model in a comparison run."""

    alias: str
    spec: ModelSpec
    campaign_result: CampaignResult | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.campaign_result is not None


class ComparisonRunner:
    """Runs identical campaigns across multiple registered models.

    Args:
        aliases: Registry aliases (e.g. ``["llama-3.1-8b", "llama-3.1-70b"]``).
        build_config: Callable ``(target_name: str) -> CampaignConfig``.
            Called once per model; used so each model gets a fresh probe set
            with the correct `target_name` stamped in.
        trajectory_log_dir: If set, per-model JSONL traces are written under
            this directory as ``{alias}.jsonl``.
        taxonomy: Optional taxonomy override (defaults to DEFAULT_TAXONOMY).
    """

    def __init__(
        self,
        aliases: Sequence[str],
        build_config,
        *,
        trajectory_log_dir: str | None = None,
        taxonomy: TechniqueRegistry | None = None,
    ) -> None:
        self.aliases = list(aliases)
        self.build_config = build_config
        self.trajectory_log_dir = trajectory_log_dir
        self.taxonomy = taxonomy or DEFAULT_TAXONOMY

    async def run(self) -> list[ModelComparisonResult]:
        results: list[ModelComparisonResult] = []
        for alias in self.aliases:
            if alias not in MODEL_REGISTRY:
                results.append(
                    ModelComparisonResult(
                        alias=alias,
                        spec=ModelSpec(
                            provider="unknown", model=alias, family="unknown"
                        ),
                        campaign_result=None,
                        error=f"Alias {alias!r} not in MODEL_REGISTRY",
                    )
                )
                continue

            spec = MODEL_REGISTRY[alias]
            logger.info(
                "Comparison: running %s (%s/%s, params_b=%s)",
                alias, spec.provider, spec.model, spec.params_b,
            )

            overrides: dict[str, Any] = {}
            if self.trajectory_log_dir:
                overrides["trajectory_log_path"] = (
                    f"{self.trajectory_log_dir.rstrip('/')}/{alias}.jsonl"
                )
            try:
                cfg = resolve(alias, **overrides)
                target: BaseTarget = create_target(cfg)
            except TargetError as exc:
                results.append(
                    ModelComparisonResult(
                        alias=alias, spec=spec, campaign_result=None,
                        error=str(exc),
                    )
                )
                continue

            campaign_config = self.build_config(alias)
            runner = CampaignRunner(
                config=campaign_config,
                target=target,
                trajectory_store=RedTeamTrajectoryStore(),
                taxonomy=self.taxonomy,
            )
            try:
                result = await runner.run()
            except Exception as exc:
                logger.exception("Comparison run failed for %s", alias)
                results.append(
                    ModelComparisonResult(
                        alias=alias, spec=spec, campaign_result=None,
                        error=str(exc),
                    )
                )
                continue

            results.append(
                ModelComparisonResult(
                    alias=alias, spec=spec, campaign_result=result, error=None
                )
            )

        return results


# ---------------------------------------------------------------------------
# Tabular aggregation (pandas-friendly, but optional)
# ---------------------------------------------------------------------------


def results_to_rows(
    results: Sequence[ModelComparisonResult],
) -> list[dict[str, Any]]:
    """Flatten `ModelComparisonResult`s to a list of dict rows.

    One row per (model, probe). Intended to be fed directly to
    ``pandas.DataFrame.from_records`` for analysis notebooks; pandas is
    imported lazily inside :func:`results_to_dataframe` so this module remains
    importable on minimal installs.
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        if not r.ok or r.campaign_result is None:
            rows.append({
                "alias": r.alias,
                "provider": r.spec.provider,
                "family": r.spec.family,
                "params_b": r.spec.params_b,
                "error": r.error,
            })
            continue
        for phase_name, probe_results in r.campaign_result.phase_results.items():
            for pr in probe_results:
                rows.append({
                    "alias": r.alias,
                    "provider": r.spec.provider,
                    "family": r.spec.family,
                    "params_b": r.spec.params_b,
                    "phase": phase_name,
                    "probe_id": pr.probe_id,
                    "technique_id": pr.technique_id,
                    "status": pr.status.value,
                    "success": pr.status == ProbeStatus.SUCCESS,
                    "confidence": pr.confidence,
                    "duration_ms": pr.duration_ms,
                })
    return rows


def results_to_dataframe(results: Sequence[ModelComparisonResult]):
    """Return a pandas DataFrame for the given comparison results."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pandas is required for results_to_dataframe; install with "
            "`pip install pandas`"
        ) from exc
    return pd.DataFrame.from_records(results_to_rows(results))


def summarize(results: Sequence[ModelComparisonResult]) -> list[dict[str, Any]]:
    """Per-model summary rows: refusal rate, success rate, mean confidence.

    This is the table format used for printing comparison reports and feeding
    the scaling regression. When StrongREJECT is wired in (Phase 2) this
    summary gains `overall_score` and `learned_overall_score` columns.
    """
    summary: list[dict[str, Any]] = []
    for r in results:
        if not r.ok or r.campaign_result is None:
            summary.append({
                "alias": r.alias,
                "family": r.spec.family,
                "params_b": r.spec.params_b,
                "log_params_b": (
                    math.log10(r.spec.params_b) if r.spec.params_b else None
                ),
                "n_probes": 0,
                "success_rate": None,
                "refusal_rate": None,
                "mean_confidence": None,
                "error": r.error,
            })
            continue
        cr = r.campaign_result
        successes = cr.total_successes
        total = cr.total_probes
        refusals = 0
        confidences: list[float] = []
        for probe_results in cr.phase_results.values():
            for pr in probe_results:
                if pr.status == ProbeStatus.FAILED:
                    refusals += 1
                confidences.append(pr.confidence)
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        summary.append({
            "alias": r.alias,
            "family": r.spec.family,
            "params_b": r.spec.params_b,
            "log_params_b": (
                math.log10(r.spec.params_b) if r.spec.params_b else None
            ),
            "n_probes": total,
            "success_rate": (successes / total) if total else 0.0,
            "refusal_rate": (refusals / total) if total else 0.0,
            "mean_confidence": mean_conf,
            "risk_score": cr.overall_risk_score,
            "duration_s": cr.duration_seconds,
            "error": None,
        })
    return summary
