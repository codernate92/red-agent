"""Evaluate existing comparison artifacts with judge LLMs (no probe rerun).

Reads a ``comparison.json`` produced by ``cli.py compare`` and runs evaluator
scoring over stored raw exchanges in each campaign result. This avoids rerunning
model probes and only performs judge-model calls.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from analysis.evaluators import StrongREJECTEvaluator, evaluate_campaign
from analysis.evaluators.evaluator_base import BaseEvaluator
from core.campaign import CampaignResult
from core.targets import create_target


def _build_evaluator(kind: str, judge_spec: str | None) -> BaseEvaluator:
    if kind != "strongreject":
        raise ValueError("Only strongreject is currently supported in evaluate_existing")

    judge = None
    if judge_spec:
        if ":" not in judge_spec:
            raise ValueError("--judge must be provider:model (e.g. openai:gpt-4o-mini)")
        provider, model = judge_spec.split(":", 1)
        judge = create_target({"provider": provider, "model": model, "temperature": 0.0})

    return StrongREJECTEvaluator(judge=judge, concurrency=4)


async def _run(args: argparse.Namespace) -> None:
    # Match CLI behavior: load environment variables from project .env.
    load_dotenv(Path(".env"))
    input_path = Path(args.input)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    payload = json.loads(input_path.read_text())
    evaluator = _build_evaluator(args.evaluate, args.judge)

    rows: list[dict[str, Any]] = []

    # Build quick index into existing summary rows so we can enrich in-place.
    by_alias = {row.get("alias"): row for row in payload.get("summary", [])}

    for pm in payload.get("per_model", []):
        alias = pm.get("alias")
        campaign_dict = pm.get("campaign")
        if not alias or not campaign_dict:
            continue

        campaign = CampaignResult.from_dict(campaign_dict)
        results = await evaluate_campaign(campaign, evaluator)
        agg = BaseEvaluator.aggregate(results)

        rows.append(
            {
                "alias": alias,
                "n": int(agg["n"]),
                "sr_refusal_rate": agg["refusal_rate"],
                "sr_mean_overall": agg["mean_overall_score"],
                "sr_mean_convincingness": agg["mean_convincingness"],
                "sr_mean_specificity": agg["mean_specificity"],
            }
        )

        # attach to summary row
        summary_row = by_alias.get(alias)
        if summary_row is not None:
            summary_row["sr_refusal_rate"] = agg["refusal_rate"]
            summary_row["sr_mean_overall"] = agg["mean_overall_score"]
            summary_row["sr_mean_convincingness"] = agg["mean_convincingness"]
            summary_row["sr_mean_specificity"] = agg["mean_specificity"]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(payload, indent=2, default=str))
    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "alias",
                "n",
                "sr_refusal_rate",
                "sr_mean_overall",
                "sr_mean_convincingness",
                "sr_mean_specificity",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote enriched comparison JSON to {output_json}")
    print(f"Wrote evaluator summary CSV to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluator on existing comparison artifact.")
    parser.add_argument("--input", required=True, help="Input comparison.json")
    parser.add_argument("--output-json", required=True, help="Output enriched comparison JSON")
    parser.add_argument("--output-csv", required=True, help="Output evaluator summary CSV")
    parser.add_argument("--evaluate", default="strongreject", choices=["strongreject"])
    parser.add_argument(
        "--judge",
        default="openai:gpt-4o-mini",
        help="Judge model spec provider:model (default: openai:gpt-4o-mini)",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
