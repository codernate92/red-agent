"""Batch vulnerability scanning across multiple registered models."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from typing import Any, Awaitable, Callable

from analysis.evaluators import StrongREJECTEvaluator
from core.model_registry import MODEL_REGISTRY, get_spec, resolve
from core.targets import BaseTarget, TargetError, create_target

from .config import ScanConfig
from .models import (
    BatchReport,
    CrossModelVulnerability,
    ModelVulnerability,
    ScalingAnalysis,
    VulnReport,
)
from .scanner import VulnScanner

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

ProviderTargetFactory = Callable[[str], BaseTarget]
EvaluatorFactory = Callable[[], Any]


class BatchScanner:
    """Run vulnerability scans across multiple registry models."""

    def __init__(
        self,
        models: list[str],
        config: ScanConfig,
        *,
        evaluator_factory: EvaluatorFactory | None = None,
        target_factory: ProviderTargetFactory | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.models = list(models)
        self.config = config
        self.evaluator_factory = evaluator_factory or StrongREJECTEvaluator
        self.target_factory = target_factory or self._default_target_factory
        self.progress_callback = progress_callback
        self._provider_limits = {
            "openai": 5,
            "anthropic": 5,
            "google": 3,
            "mistral": 3,
            "together": 3,
            "ollama": 2,
            "vllm": 2,
        }

    async def scan_all(self) -> BatchReport:
        """Scan all models in parallel with provider-aware concurrency."""
        semaphores: dict[str, asyncio.Semaphore] = {}
        for alias in self.models:
            provider = MODEL_REGISTRY.get(alias).provider if alias in MODEL_REGISTRY else "unknown"
            semaphores.setdefault(
                provider,
                asyncio.Semaphore(
                    min(
                        self.config.max_concurrent_requests,
                        self._provider_limits.get(provider, self.config.max_concurrent_requests),
                    )
                ),
            )

        results = await asyncio.gather(
            *[
                asyncio.create_task(self._scan_one(alias, semaphores))
                for alias in self.models
            ]
        )

        reports: dict[str, VulnReport] = {}
        errors: dict[str, str] = {}
        for alias, report, error in results:
            if error is not None or report is None:
                errors[alias] = error or "unknown error"
            else:
                reports[alias] = report

        return BatchReport(
            reports=reports,
            errors=errors,
            cross_model_comparison=self._cross_model_comparison(reports),
            scaling_analysis=self._scaling_analysis(reports),
            worst_findings=self._worst_findings(reports),
        )

    async def _scan_one(
        self,
        alias: str,
        semaphores: dict[str, asyncio.Semaphore],
    ) -> tuple[str, VulnReport | None, str | None]:
        if alias not in MODEL_REGISTRY:
            return alias, None, f"Unknown registry alias: {alias}"

        provider = MODEL_REGISTRY[alias].provider
        semaphore = semaphores[provider]
        try:
            target = self.target_factory(alias)
            evaluator = self.evaluator_factory()
            scanner = VulnScanner(
                target,
                evaluator,
                self.config,
                model_name=alias,
                request_semaphore=semaphore,
                progress_callback=self._wrap_progress(alias, provider),
            )
            report = await scanner.scan()
            return alias, report, None
        except (TargetError, KeyError, ValueError) as exc:
            logger.exception("Batch scan failed for %s", alias)
            return alias, None, str(exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected batch scan failure for %s", alias)
            return alias, None, str(exc)

    def _default_target_factory(self, alias: str) -> BaseTarget:
        cfg = resolve(alias)
        return create_target(cfg)

    def _wrap_progress(self, alias: str, provider: str) -> ProgressCallback | None:
        if self.progress_callback is None:
            return None

        async def _callback(payload: dict[str, Any]) -> None:
            merged = dict(payload)
            merged["alias"] = alias
            merged["provider"] = provider
            result = self.progress_callback(merged)
            if inspect.isawaitable(result):
                await result

        return _callback

    def _cross_model_comparison(
        self,
        reports: dict[str, VulnReport],
    ) -> dict[str, list[CrossModelVulnerability]]:
        if not reports:
            return {"universal": [], "model_specific": []}

        signatures: dict[str, list[tuple[str, float]]] = {}
        for alias, report in reports.items():
            for vulnerability in report.vulnerabilities:
                signature = f"{vulnerability.attack_type}::{vulnerability.technique}"
                signatures.setdefault(signature, []).append((alias, vulnerability.severity))

        universal: list[CrossModelVulnerability] = []
        model_specific: list[CrossModelVulnerability] = []
        model_count = len(reports)

        for signature, rows in signatures.items():
            models = sorted(alias for alias, _ in rows)
            severities = [severity for _, severity in rows]
            item = CrossModelVulnerability(
                signature=signature,
                models=models,
                mean_severity=sum(severities) / len(severities),
                max_severity=max(severities),
            )
            if len(models) == model_count:
                universal.append(item)
            elif len(models) == 1:
                model_specific.append(item)

        universal.sort(key=lambda item: item.max_severity, reverse=True)
        model_specific.sort(key=lambda item: item.max_severity, reverse=True)
        return {"universal": universal, "model_specific": model_specific}

    def _scaling_analysis(
        self,
        reports: dict[str, VulnReport],
    ) -> ScalingAnalysis | None:
        points: list[tuple[float, float]] = []
        for alias, report in reports.items():
            spec = get_spec(alias)
            if spec.params_b is None or spec.params_b <= 0:
                continue
            worst = report.vulnerabilities[0].severity if report.vulnerabilities else 0.0
            points.append((math.log10(spec.params_b), worst))

        if len(points) < 2:
            return None

        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom == 0:
            return None
        slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
        intercept = mean_y - slope * mean_x
        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot else 1.0

        return ScalingAnalysis(
            metric="worst_severity",
            sample_size=len(points),
            slope=round(slope, 4),
            intercept=round(intercept, 4),
            r_squared=round(max(0.0, min(1.0, r_squared)), 4),
        )

    def _worst_findings(self, reports: dict[str, VulnReport]) -> list[ModelVulnerability]:
        flattened: list[ModelVulnerability] = []
        for alias, report in reports.items():
            for vulnerability in report.vulnerabilities:
                flattened.append(
                    ModelVulnerability(
                        model=alias,
                        vulnerability=vulnerability,
                    )
                )
        flattened.sort(
            key=lambda item: item.vulnerability.severity,
            reverse=True,
        )
        return flattened[:10]
