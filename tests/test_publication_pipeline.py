from __future__ import annotations

import json

import pytest

from analysis import provider_status
from analysis.publication_report import build_publication_report
from core.targets.base_target import BaseTarget, TargetError


class AlwaysFailTarget(BaseTarget):
    provider = "fake"

    async def _aquery(self, messages, **kwargs):  # type: ignore[override]
        raise ValueError("synthetic failure")


@pytest.mark.asyncio
async def test_base_target_logs_failed_requests(tmp_path) -> None:
    log_path = tmp_path / "failed.jsonl"
    target = AlwaysFailTarget(
        model="fake-model",
        trajectory_log_path=log_path,
        max_retries=1,
    )

    with pytest.raises(TargetError):
        await target.query("hello")

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "error"
    assert row["provider"] == "fake"
    assert row["model"] == "fake-model"
    assert row["response"] is None
    assert row["error_type"] == "ValueError"
    assert "synthetic failure" in row["error"]


@pytest.mark.asyncio
async def test_provider_status_failure_rows_include_registry_metadata(monkeypatch) -> None:
    class FailingProbe:
        provider = "ignored"
        model = "ignored"

        async def query(self, prompt: str):  # noqa: ARG002
            raise RuntimeError("quota boom")

    monkeypatch.setattr(provider_status, "create_target", lambda cfg: FailingProbe())

    row = await provider_status.probe_alias("gpt-4o-mini")

    assert row["ok"] is False
    assert row["provider"] == "openai"
    assert row["model"] == "gpt-4o-mini"
    assert row["family"] == "gpt"
    assert row["source"] == "preflight_probe"
    assert "quota boom" in row["error"]


def test_publication_report_includes_error_rates_and_disclosure(tmp_path) -> None:
    comparison_payload = {
        "campaign": "full",
        "evaluator": "strongreject",
        "judge": "anthropic:claude-sonnet-4-6",
        "summary": [
            {
                "alias": "mistral-small",
                "family": "mistral",
                "params_b": 22.0,
                "n_probes": 2,
                "success_rate": 0.5,
                "refusal_rate": 0.0,
                "risk_score": 3.7,
                "sr_mean_overall": 0.10,
                "sr_refusal_rate": 0.5,
                "error": None,
            }
        ],
        "per_model": [
            {
                "alias": "mistral-small",
                "campaign": {
                    "phase_results": {
                        "phase": [
                            {
                                "technique_id": "T1004",
                                "status": "success",
                                "confidence": 0.8,
                            },
                            {
                                "technique_id": "T1003",
                                "status": "error",
                                "confidence": 0.0,
                            },
                        ]
                    }
                },
                "evaluations": [
                    {"technique_id": "T1004", "overall_score": 0.20},
                    {"technique_id": "T1003", "overall_score": 0.00},
                ],
            }
        ],
    }
    status_payload = [
        {
            "spec": "mistral-small",
            "provider": "mistral",
            "model": "mistral-small-latest",
            "family": "mistral",
            "ok": True,
            "source": "compare_run",
        },
        {
            "spec": "gpt-4o-mini",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "family": "gpt",
            "ok": False,
            "error": "insufficient quota",
            "source": "preflight_probe",
        },
    ]

    comparison_path = tmp_path / "comparison.json"
    status_path = tmp_path / "status.json"
    output_dir = tmp_path / "publication"
    comparison_path.write_text(json.dumps(comparison_payload))
    status_path.write_text(json.dumps(status_payload))

    artifacts = build_publication_report(
        comparison_path,
        output_dir,
        status_json=status_path,
    )

    assert artifacts["report_md"].exists()
    assert artifacts["report_tex"].exists()
    assert artifacts["model_summary_tex"].exists()

    markdown = artifacts["report_md"].read_text()
    assert "Error rate" in markdown
    assert "Authorship And Tooling Disclosure" in markdown

    tex = artifacts["report_tex"].read_text()
    assert "Authorship and Tooling Disclosure" in tex
    assert "Only the mistral family contributed reachable size-annotated points" in tex

    summary_csv = (output_dir / "summary.csv").read_text()
    assert "error_rate" in summary_csv.splitlines()[0]

    model_table = artifacts["model_summary_tex"].read_text()
    assert "Error rate" in model_table
