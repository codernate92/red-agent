"""Generate publication-style report assets from comparison output."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from core.taxonomy import DEFAULT_TAXONOMY


plt.style.use("seaborn-v0_8-whitegrid")


def build_publication_report(
    comparison_json: str | Path,
    output_dir: str | Path,
    *,
    status_json: str | Path | None = None,
) -> dict[str, Path]:
    """Build CSV, plots, and a Markdown summary from comparison output."""
    comparison_path = Path(comparison_json)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(comparison_path.read_text())
    summary_rows: list[dict[str, Any]] = list(payload.get("summary", []))
    per_model: list[dict[str, Any]] = list(payload.get("per_model", []))
    status_rows: list[dict[str, Any]] = []
    if status_json is not None:
        status_rows = list(json.loads(Path(status_json).read_text()))

    summary_csv = output_path / "summary.csv"
    _write_csv(summary_csv, summary_rows)

    technique_rows = _build_technique_rows(per_model)
    technique_csv = output_path / "technique_success.csv"
    _write_csv(technique_csv, technique_rows)

    artifacts: dict[str, Path] = {
        "summary_csv": summary_csv,
        "technique_csv": technique_csv,
        "report_md": output_path / "publication_report.md",
        "risk_bar_png": plots_dir / "risk_bar.png",
        "metric_heatmap_png": plots_dir / "metric_heatmap.png",
        "technique_heatmap_png": plots_dir / "technique_heatmap.png",
        "scaling_plot_png": plots_dir / "scaling_plot.png",
    }

    if status_rows:
        status_csv = output_path / "provider_status.csv"
        _write_csv(status_csv, status_rows)
        artifacts["status_csv"] = status_csv
        artifacts["provider_status_png"] = plots_dir / "provider_status.png"

    _plot_risk_bar(summary_rows, artifacts["risk_bar_png"])
    _plot_metric_heatmap(summary_rows, artifacts["metric_heatmap_png"])
    _plot_technique_heatmap(technique_rows, artifacts["technique_heatmap_png"])
    _plot_scaling(summary_rows, artifacts["scaling_plot_png"])
    if status_rows:
        _plot_provider_status(status_rows, artifacts["provider_status_png"])
    _write_markdown_report(
        payload,
        summary_rows,
        technique_rows,
        status_rows,
        artifacts["report_md"],
        artifacts,
    )

    return artifacts


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_technique_rows(per_model: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in per_model:
        alias = item["alias"]
        campaign = item.get("campaign")
        if not campaign:
            continue
        phase_results = campaign.get("phase_results", {})
        by_technique: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for results in phase_results.values():
            for result in results:
                by_technique[result["technique_id"]].append(result)
        evals = item.get("evaluations", [])
        eval_by_technique: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ev in evals:
            eval_by_technique[ev.get("technique_id", "unknown")].append(ev)

        for technique_id, results in by_technique.items():
            total = len(results)
            successes = sum(1 for result in results if result["status"] == "success")
            mean_conf = (
                sum(float(result.get("confidence", 0.0)) for result in results) / total
                if total
                else 0.0
            )
            eval_bucket = eval_by_technique.get(technique_id, [])
            mean_overall = (
                sum(float(ev.get("overall_score", 0.0)) for ev in eval_bucket) / len(eval_bucket)
                if eval_bucket
                else 0.0
            )
            try:
                technique = DEFAULT_TAXONOMY.get(technique_id)
                technique_name = technique.name
                tactic = technique.tactic.value
            except KeyError:
                technique_name = technique_id
                tactic = "unknown"
            rows.append(
                {
                    "alias": alias,
                    "technique_id": technique_id,
                    "technique_name": technique_name,
                    "tactic": tactic,
                    "success_rate": successes / total if total else 0.0,
                    "mean_confidence": mean_conf,
                    "mean_overall_score": mean_overall,
                    "n": total,
                }
            )
    return rows


def _plot_risk_bar(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    rows = [row for row in summary_rows if not row.get("error")]
    if not rows:
        return

    rows.sort(key=lambda row: float(row.get("risk_score", 0.0)), reverse=True)
    labels = [row["alias"] for row in rows]
    values = [float(row.get("risk_score", 0.0)) for row in rows]

    fig, ax = plt.subplots(figsize=(13, max(6, 0.42 * len(labels))))
    bars = ax.barh(labels, values, color="#b63a32")
    ax.invert_yaxis()
    ax.set_xlabel("Risk Score")
    ax.set_title("Overall Risk Ranking")
    ax.set_xlim(0, max(10.0, max(values) + 0.6))
    for bar, value in zip(bars, values):
        ax.text(value + 0.08, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9)
    _save_figure(fig, output_path)


def _plot_metric_heatmap(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    rows = [row for row in summary_rows if not row.get("error")]
    if not rows:
        return

    rows.sort(key=lambda row: float(row.get("risk_score", 0.0)), reverse=True)
    labels = [row["alias"] for row in rows]
    metric_names = [
        ("risk_score", "Risk"),
        ("success_rate", "Hit Rate"),
        ("refusal_rate", "Probe Refusal"),
        ("sr_mean_overall", "SR Overall"),
        ("sr_refusal_rate", "SR Refusal"),
    ]
    matrix = [
        [float(row.get(metric, 0.0) or 0.0) for metric, _ in metric_names]
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.48 * len(labels))))
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=max(1.0, max(max(r) for r in matrix)))
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels([label for _, label in metric_names], rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Cross-Model Metric Heatmap")
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Score")
    _save_figure(fig, output_path)


def _plot_technique_heatmap(technique_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not technique_rows:
        return

    aliases = sorted({row["alias"] for row in technique_rows})
    ranked_techniques = sorted(
        {
            row["technique_name"]: float(row.get("mean_overall_score", 0.0))
            for row in technique_rows
        }.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    techniques = [name for name, _ in ranked_techniques[:15]]
    index = {
        (row["alias"], row["technique_name"]): float(row.get("mean_overall_score", 0.0))
        for row in technique_rows
        if row["technique_name"] in techniques
    }
    matrix = [
        [index.get((alias, technique), 0.0) for technique in techniques]
        for alias in aliases
    ]

    fig, ax = plt.subplots(figsize=(max(12, 0.75 * len(techniques)), max(6, 0.45 * len(aliases))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(techniques)))
    ax.set_xticklabels(techniques, rotation=55, ha="right")
    ax.set_yticks(range(len(aliases)))
    ax.set_yticklabels(aliases)
    ax.set_title("Top Technique Heatmap")
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Overall Score")
    _save_figure(fig, output_path)


def _plot_scaling(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    rows = [
        row
        for row in summary_rows
        if not row.get("error") and row.get("params_b") not in (None, "", 0)
    ]
    if not rows:
        return

    families = sorted({row["family"] for row in rows})
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#bcbd22"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for index, family in enumerate(families):
        family_rows = [row for row in rows if row["family"] == family]
        family_rows.sort(key=lambda row: float(row["params_b"]))
        xs = [math.log10(float(row["params_b"])) for row in family_rows]
        ys = [float(row.get("sr_mean_overall", row.get("risk_score", 0.0))) for row in family_rows]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            label=family,
            color=palette[index % len(palette)],
        )
        for x_value, y_value, row in zip(xs, ys, family_rows):
            ax.text(x_value, y_value + 0.015, row["alias"], fontsize=8, ha="center")

    ax.set_xlabel("log10(parameters in billions)")
    ax.set_ylabel("StrongREJECT overall score")
    ax.set_title("Scaling Trend by Model Family")
    ax.legend(frameon=False, fontsize=9)
    _save_figure(fig, output_path)


def _plot_provider_status(status_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not status_rows:
        return

    rows = list(status_rows)
    rows.sort(key=lambda row: (not bool(row.get("ok")), row.get("spec", "")))
    labels = [row["spec"] for row in rows]
    values = [1 if row.get("ok") else 0 for row in rows]
    colors = ["#2e8b57" if value else "#b63a32" for value in values]

    fig, ax = plt.subplots(figsize=(13, max(5, 0.42 * len(labels))))
    bars = ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Reachability")
    ax.set_title("Live Provider Reachability")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["blocked", "ok"])
    for bar, row in zip(bars, rows):
        label = "ok" if row.get("ok") else str(row.get("error", ""))[:90]
        ax.text(1.02 if row.get("ok") else 0.02, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8)
    _save_figure(fig, output_path)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _write_markdown_report(
    payload: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    technique_rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    output_path: Path,
    artifacts: dict[str, Path],
) -> None:
    valid_rows = [row for row in summary_rows if not row.get("error")]
    failed_rows = [row for row in summary_rows if row.get("error")]
    valid_rows.sort(key=lambda row: float(row.get("risk_score", 0.0)), reverse=True)

    top_risk = valid_rows[:5]
    top_sr = sorted(
        valid_rows,
        key=lambda row: float(row.get("sr_mean_overall", 0.0) or 0.0),
        reverse=True,
    )[:5]

    technique_strength = defaultdict(list)
    for row in technique_rows:
        technique_strength[row["technique_name"]].append(float(row.get("mean_overall_score", 0.0)))
    top_techniques = sorted(
        (
            (name, sum(scores) / len(scores))
            for name, scores in technique_strength.items()
            if scores
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    blocked_status_rows = [row for row in status_rows if not row.get("ok")]
    reachable_status_rows = [row for row in status_rows if row.get("ok")]

    lines: list[str] = []
    lines.append("# Publication Report")
    lines.append("")
    lines.append(f"- Campaign: `{payload.get('campaign', 'unknown')}`")
    lines.append(f"- Evaluator: `{payload.get('evaluator', 'none')}`")
    lines.append(f"- Judge: `{payload.get('judge') or 'default'}`")
    lines.append(f"- Models completed: {len(valid_rows)} / {len(summary_rows)}")
    if status_rows:
        lines.append(f"- Providers reachable in live preflight: {len(reachable_status_rows)} / {len(status_rows)}")
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    if top_risk:
        strongest = top_risk[0]
        lines.append(
            f"- Highest measured risk was `{strongest['alias']}` at `{float(strongest.get('risk_score', 0.0)):.2f}` "
            f"with StrongREJECT overall `{float(strongest.get('sr_mean_overall', 0.0) or 0.0):.3f}`."
        )
    if top_techniques:
        lines.append(
            f"- The most effective technique across completed runs was `{top_techniques[0][0]}` "
            f"with mean overall score `{top_techniques[0][1]:.3f}`."
        )
    if blocked_status_rows:
        lines.append(
            f"- External execution blockers affected {len(blocked_status_rows)} configured providers or model endpoints; "
            "those are listed explicitly below instead of being treated as benign zeros."
        )
    lines.append("")

    if failed_rows or blocked_status_rows:
        lines.append("## Execution Blockers")
        lines.append("")
        if failed_rows:
            lines.append("| Model | Family | Error |")
            lines.append("|-------|--------|-------|")
            for row in failed_rows:
                error = str(row.get("error", "")).replace("\n", " ").replace("|", "/")
                lines.append(f"| {row['alias']} | {row.get('family', '-')} | {error[:180]} |")
            lines.append("")
        if blocked_status_rows:
            lines.append("| Preflight Spec | Status | Detail |")
            lines.append("|---------------|--------|--------|")
            for row in blocked_status_rows:
                detail = str(row.get("error", "")).replace("\n", " ").replace("|", "/")
                lines.append(f"| {row['spec']} | blocked | {detail[:180]} |")
            lines.append("")

    lines.append("## Artifact Bundle")
    lines.append("")
    for label, path in artifacts.items():
        rel = path.relative_to(output_path.parent)
        lines.append(f"- `{label}`: `{rel}`")
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    for key in (
        "risk_bar_png",
        "metric_heatmap_png",
        "technique_heatmap_png",
        "scaling_plot_png",
        "provider_status_png",
    ):
        if key in artifacts:
            rel = artifacts[key].relative_to(output_path.parent)
            lines.append(f"### {key.replace('_png', '').replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"![]({rel.as_posix()})")
            lines.append("")

    lines.append("## Top Risk Scores")
    lines.append("")
    lines.append("| Model | Family | Risk | Hit Rate | SR Overall | SR Refusal |")
    lines.append("|-------|--------|------|----------|------------|------------|")
    for row in top_risk:
        lines.append(
            f"| {row['alias']} | {row['family']} | {float(row.get('risk_score', 0.0)):.2f} | "
            f"{float(row.get('success_rate', 0.0) or 0.0):.1%} | "
            f"{float(row.get('sr_mean_overall', 0.0) or 0.0):.3f} | "
            f"{float(row.get('sr_refusal_rate', 0.0) or 0.0):.1%} |"
        )
    lines.append("")

    lines.append("## Highest Evaluator Scores")
    lines.append("")
    lines.append("| Model | Family | SR Overall | Risk | Params (B) |")
    lines.append("|-------|--------|------------|------|------------|")
    for row in top_sr:
        lines.append(
            f"| {row['alias']} | {row['family']} | {float(row.get('sr_mean_overall', 0.0) or 0.0):.3f} | "
            f"{float(row.get('risk_score', 0.0)):.2f} | {row.get('params_b', '-')} |"
        )
    lines.append("")

    lines.append("## Most Effective Techniques")
    lines.append("")
    lines.append("| Technique | Mean Overall Score |")
    lines.append("|-----------|--------------------|")
    for name, score in top_techniques:
        lines.append(f"| {name} | {score:.3f} |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This artifact bundle is intended for publication-quality review: it includes "
        "annotated heatmaps, vector exports, a scaling plot, tabular CSV exports, and an "
        "explicit separation between security findings and provider/account execution blockers."
    )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-style report assets from comparison JSON.")
    parser.add_argument("--input", required=True, help="Path to comparison JSON output.")
    parser.add_argument("--output-dir", required=True, help="Directory for report artifacts.")
    parser.add_argument(
        "--status-json",
        default=None,
        help="Optional path to provider preflight status JSON.",
    )
    args = parser.parse_args()
    build_publication_report(args.input, args.output_dir, status_json=args.status_json)


if __name__ == "__main__":
    main()
