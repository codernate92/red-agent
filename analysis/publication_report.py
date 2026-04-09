"""Generate publication-grade report assets from comparison output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

from core.model_registry import MODEL_REGISTRY
from core.taxonomy import DEFAULT_TAXONOMY


BACKGROUND = "#f3efe8"
PANEL = "#fbf8f2"
INK = "#201e1d"
GRID = "#d8d1c7"
ACCENT = "#b14a32"
ACCENT_2 = "#275d8f"
ACCENT_3 = "#5b8f4e"
BLOCKED = "#c84b44"
OK = "#4f8f66"

FAMILY_COLORS = {
    "claude": "#7b4ea3",
    "mistral": "#cf5c36",
    "gpt": "#2b6cb0",
    "gpt-oss": "#2f855a",
    "gemini": "#c28b1d",
    "llama-3.1": "#1f7a8c",
    "gemma-2": "#6a4c93",
    "mixtral": "#9a3412",
}

FIGURE_FORMATS = (".png", ".pdf", ".svg")
DISCLOSURE_TEXT = (
    "This project used AI assistance for portions of implementation, figure "
    "design, and manuscript drafting. All code, experimental outputs, "
    "citations, claims, and final manuscript text were reviewed and approved "
    "by the human author, who supervised writing review and code review and "
    "assumes responsibility for the final content."
)


mpl.rcParams.update(
    {
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BACKGROUND,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "axes.labelsize": 11,
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "grid.color": GRID,
        "grid.alpha": 0.45,
        "axes.grid": False,
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "legend.frameon": False,
        "legend.fontsize": 9,
    }
)


def build_publication_report(
    comparison_json: str | Path,
    output_dir: str | Path,
    *,
    status_json: str | Path | None = None,
) -> dict[str, Path]:
    """Build LaTeX-ready plots, tables, and report sources."""
    comparison_path = Path(comparison_json)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plots_dir = output_path / "plots"
    tables_dir = output_path / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(comparison_path.read_text())
    summary_rows: list[dict[str, Any]] = list(payload.get("summary", []))
    per_model: list[dict[str, Any]] = list(payload.get("per_model", []))
    status_rows: list[dict[str, Any]] = []
    if status_json is not None:
        status_rows = list(json.loads(Path(status_json).read_text()))

    summary_rows = _enrich_summary_rows(summary_rows, per_model)
    valid_rows = [row for row in summary_rows if not row.get("error")]
    valid_rows.sort(key=lambda row: float(row.get("risk_score", 0.0)), reverse=True)
    technique_rows = _build_technique_rows(per_model)

    summary_csv = output_path / "summary.csv"
    technique_csv = output_path / "technique_success.csv"
    _write_csv(summary_csv, summary_rows)
    _write_csv(technique_csv, technique_rows)

    artifacts: dict[str, Path] = {
        "summary_csv": summary_csv,
        "technique_csv": technique_csv,
        "report_md": output_path / "publication_report.md",
        "report_tex": output_path / "report.tex",
        "overview_dashboard_pdf": plots_dir / "overview_dashboard.pdf",
        "risk_bar_pdf": plots_dir / "risk_bar.pdf",
        "frontier_pdf": plots_dir / "frontier.pdf",
        "metric_heatmap_pdf": plots_dir / "metric_heatmap.pdf",
        "technique_heatmap_pdf": plots_dir / "technique_heatmap.pdf",
        "scaling_plot_pdf": plots_dir / "scaling_plot.pdf",
    }
    if status_rows:
        status_csv = output_path / "provider_status.csv"
        _write_csv(status_csv, status_rows)
        artifacts["status_csv"] = status_csv
        artifacts["provider_status_pdf"] = plots_dir / "provider_status.pdf"

    _plot_overview_dashboard(valid_rows, technique_rows, status_rows, artifacts["overview_dashboard_pdf"])
    _plot_risk_bar(valid_rows, artifacts["risk_bar_pdf"])
    _plot_frontier(valid_rows, artifacts["frontier_pdf"])
    _plot_metric_heatmap(valid_rows, artifacts["metric_heatmap_pdf"])
    _plot_technique_heatmap(technique_rows, artifacts["technique_heatmap_pdf"])
    _plot_scaling(valid_rows, artifacts["scaling_plot_pdf"])
    if status_rows:
        _plot_provider_status(status_rows, artifacts["provider_status_pdf"])

    model_table = tables_dir / "model_summary.tex"
    technique_table = tables_dir / "top_techniques.tex"
    blocker_table = tables_dir / "execution_blockers.tex"
    _write_model_summary_table(valid_rows, model_table)
    _write_top_techniques_table(technique_rows, technique_table)
    _write_blockers_table(status_rows, blocker_table)

    artifacts["model_summary_tex"] = model_table
    artifacts["top_techniques_tex"] = technique_table
    artifacts["execution_blockers_tex"] = blocker_table

    _write_markdown_report(
        payload,
        valid_rows,
        technique_rows,
        status_rows,
        artifacts["report_md"],
        artifacts,
    )
    _write_tex_report(
        payload,
        valid_rows,
        technique_rows,
        status_rows,
        artifacts["report_tex"],
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
        eval_by_technique: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ev in item.get("evaluations", []):
            eval_by_technique[ev.get("technique_id", "unknown")].append(ev)

        for technique_id, results in by_technique.items():
            total = len(results)
            successes = sum(1 for result in results if result["status"] == "success")
            mean_conf = sum(float(result.get("confidence", 0.0)) for result in results) / total if total else 0.0
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


def _enrich_summary_rows(
    summary_rows: list[dict[str, Any]],
    per_model: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Backfill operational rates from raw per-model phase results."""
    per_model_index = {item["alias"]: item for item in per_model}
    enriched: list[dict[str, Any]] = []

    for row in summary_rows:
        item = per_model_index.get(row["alias"])
        counts: Counter[str] = Counter()
        total = int(row.get("n_probes", 0) or 0)
        if item and item.get("campaign"):
            total = 0
            phase_results = item["campaign"].get("phase_results", {})
            for results in phase_results.values():
                for result in results:
                    status = str(result.get("status", "unknown"))
                    counts[status] += 1
                    total += 1

        enriched_row = dict(row)
        if total:
            successes = counts.get("success", int(row.get("n_successes", 0) or 0))
            failures = counts.get("failed", int(row.get("n_failures", 0) or 0))
            errors = counts.get("error", int(row.get("n_errors", 0) or 0))
            partials = counts.get("partial", int(row.get("n_partials", 0) or 0))
            enriched_row.update(
                {
                    "n_probes": total,
                    "n_successes": successes,
                    "n_failures": failures,
                    "n_errors": errors,
                    "n_partials": partials,
                    "success_rate": float(enriched_row.get("success_rate", successes / total if total else 0.0) or 0.0),
                    "refusal_rate": float(enriched_row.get("refusal_rate", failures / total if total else 0.0) or 0.0),
                    "error_rate": errors / total,
                    "partial_rate": partials / total,
                    "completed_rate": (total - errors) / total,
                }
            )
        enriched.append(enriched_row)

    return enriched


def _status_provider(row: dict[str, Any]) -> str:
    provider = row.get("provider")
    if provider:
        return str(provider)
    spec = str(row.get("spec", "unknown"))
    if spec in MODEL_REGISTRY:
        return MODEL_REGISTRY[spec].provider
    return "unknown"


def _scaling_note(summary_rows: list[dict[str, Any]]) -> str:
    rows = [row for row in summary_rows if row.get("params_b") not in (None, "", 0)]
    families = sorted({str(row.get("family", "unknown")) for row in rows})
    if not rows:
        return "No reachable models in this run had known parameter counts, so no scaling interpretation is supported."
    if len(families) == 1:
        return (
            f"Only the {families[0]} family contributed reachable size-annotated points, "
            "so the size trend is exploratory rather than a general scaling result."
        )
    return (
        "The scaling plot should still be treated as exploratory because multiple configured "
        "providers were blocked by quota, credentials, or stale model identifiers."
    )


def _top_techniques(technique_rows: list[dict[str, Any]], limit: int = 10) -> list[tuple[str, float]]:
    scores: dict[str, list[float]] = defaultdict(list)
    for row in technique_rows:
        scores[row["technique_name"]].append(float(row.get("mean_overall_score", 0.0)))
    ranked = sorted(
        ((name, sum(vals) / len(vals)) for name, vals in scores.items() if vals),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:limit]


def _family_color(family: str) -> str:
    return FAMILY_COLORS.get(family, ACCENT_2)


def _wrap_label(text: str, width: int = 20) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def _outline_text(text_obj: Any) -> None:
    text_obj.set_path_effects([patheffects.withStroke(linewidth=3, foreground=PANEL)])


def _plot_overview_dashboard(
    summary_rows: list[dict[str, Any]],
    technique_rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(15.5, 10.2), constrained_layout=True)
    mosaic = fig.subplot_mosaic(
        [["risk", "frontier"], ["techniques", "status"]],
        gridspec_kw={"width_ratios": [1.15, 1.0], "height_ratios": [1.0, 1.0]},
    )

    risk_ax = mosaic["risk"]
    frontier_ax = mosaic["frontier"]
    techniques_ax = mosaic["techniques"]
    status_ax = mosaic["status"]

    rows = list(summary_rows)
    rows.sort(key=lambda row: float(row.get("risk_score", 0.0)), reverse=True)
    labels = [row["alias"] for row in rows]
    values = [float(row.get("risk_score", 0.0)) for row in rows]
    colors = [_family_color(str(row.get("family", ""))) for row in rows]
    bars = risk_ax.barh(labels, values, color=colors, alpha=0.95)
    risk_ax.invert_yaxis()
    risk_ax.set_title("Risk Ranking")
    risk_ax.set_xlabel("CVSS-style risk score")
    risk_ax.set_xlim(0, max(5.0, max(values, default=0.0) + 0.75))
    risk_ax.xaxis.grid(True, color=GRID)
    for bar, row in zip(bars, rows):
        label = f"{float(row.get('risk_score', 0.0)):.2f} | SR {float(row.get('sr_mean_overall', 0.0) or 0.0):.3f}"
        text = risk_ax.text(
            bar.get_width() + 0.07,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8.5,
            color=INK,
        )
        _outline_text(text)

    _draw_frontier(frontier_ax, summary_rows, title="Attack Frontier")

    top_techniques = _top_techniques(technique_rows, limit=8)
    tech_labels = [_wrap_label(name, 28) for name, _ in top_techniques]
    tech_values = [score for _, score in top_techniques]
    tech_bars = techniques_ax.barh(tech_labels, tech_values, color=ACCENT_2, alpha=0.9)
    techniques_ax.invert_yaxis()
    techniques_ax.set_title("Most Effective Techniques")
    techniques_ax.set_xlabel("Mean StrongREJECT overall score")
    techniques_ax.set_xlim(0, max(0.35, max(tech_values, default=0.0) + 0.03))
    techniques_ax.xaxis.grid(True, color=GRID)
    for bar, score in zip(tech_bars, tech_values):
        text = techniques_ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=8.5)
        _outline_text(text)

    _draw_provider_status(status_ax, status_rows, title="Registry Reachability")

    fig.suptitle("LLM Vulnerability Scan: Executive Dashboard", fontsize=20, fontweight="bold", y=1.02)
    _save_figure(fig, output_path)


def _draw_frontier(ax: Any, summary_rows: list[dict[str, Any]], *, title: str) -> None:
    rows = list(summary_rows)
    xs = [1.0 - float(row.get("sr_refusal_rate", 0.0) or 0.0) for row in rows]
    ys = [float(row.get("sr_mean_overall", 0.0) or 0.0) for row in rows]
    sizes = [250 + 700 * (float(row.get("risk_score", 0.0)) / 10.0) for row in rows]
    colors = [_family_color(str(row.get("family", ""))) for row in rows]

    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.88, edgecolors=BACKGROUND, linewidths=1.8)
    for row, x_value, y_value in zip(rows, xs, ys):
        text = ax.text(x_value + 0.01, y_value + 0.002, row["alias"], fontsize=8.5)
        _outline_text(text)

    ax.set_title(title)
    ax.set_xlabel("Target-model compliance rate (1 - judge-scored refusal)")
    ax.set_ylabel("StrongREJECT overall score")
    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(0.0, max(0.15, max(ys, default=0.0) + 0.025))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.grid(True, color=GRID)
    ax.xaxis.grid(True, color=GRID)


def _plot_risk_bar(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    rows = list(summary_rows)
    if not rows:
        return
    rows.sort(key=lambda row: float(row.get("risk_score", 0.0)), reverse=True)
    labels = [row["alias"] for row in rows]
    values = [float(row.get("risk_score", 0.0)) for row in rows]
    colors = [_family_color(str(row.get("family", ""))) for row in rows]

    fig, ax = plt.subplots(figsize=(12.8, max(5.4, 0.58 * len(labels))))
    bars = ax.barh(labels, values, color=colors, alpha=0.95)
    ax.invert_yaxis()
    ax.set_xlabel("Risk score")
    ax.set_title("Overall Vulnerability Ranking")
    ax.set_xlim(0, max(5.0, max(values, default=0.0) + 0.8))
    ax.xaxis.grid(True, color=GRID)
    for bar, row in zip(bars, rows):
        label = f"hit {float(row.get('success_rate', 0.0) or 0.0):.0%} | SR {float(row.get('sr_mean_overall', 0.0) or 0.0):.3f}"
        text = ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9)
        _outline_text(text)
    _save_figure(fig, output_path)


def _plot_frontier(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    _draw_frontier(ax, summary_rows, title="Vulnerability Frontier")
    _save_figure(fig, output_path)


def _plot_metric_heatmap(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    rows = list(summary_rows)
    if not rows:
        return
    rows.sort(key=lambda row: float(row.get("risk_score", 0.0)), reverse=True)
    labels = [row["alias"] for row in rows]

    normalized = []
    annotations: list[list[str]] = []
    for row in rows:
        risk = float(row.get("risk_score", 0.0)) / 10.0
        hit_rate = float(row.get("success_rate", 0.0) or 0.0)
        permissive_probe = 1.0 - float(row.get("refusal_rate", 0.0) or 0.0)
        sr_score = float(row.get("sr_mean_overall", 0.0) or 0.0)
        permissive_judge = 1.0 - float(row.get("sr_refusal_rate", 0.0) or 0.0)
        normalized.append([risk, hit_rate, permissive_probe, sr_score, permissive_judge])
        annotations.append(
            [
                f"{float(row.get('risk_score', 0.0)):.2f}",
                f"{hit_rate:.0%}",
                f"{permissive_probe:.0%}",
                f"{sr_score:.3f}",
                f"{permissive_judge:.0%}",
            ]
        )

    cmap = LinearSegmentedColormap.from_list("paper_heat", ["#f6f2eb", "#f2c48d", "#cf5c36", "#5d174a"])
    fig, ax = plt.subplots(figsize=(10.6, max(5.2, 0.55 * len(labels))))
    im = ax.imshow(normalized, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Risk / 10", "Hit rate", "Probe comply", "SR overall", "Judge comply"], rotation=25, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Normalized Vulnerability Heatmap")
    for row_idx, row in enumerate(annotations):
        for col_idx, value in enumerate(row):
            text = ax.text(col_idx, row_idx, value, ha="center", va="center", fontsize=8.5, color="white")
            text.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="#00000066")])
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Normalized vulnerability intensity")
    _save_figure(fig, output_path)


def _plot_technique_heatmap(technique_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not technique_rows:
        return
    aliases = sorted({row["alias"] for row in technique_rows})
    ranked = _top_techniques(technique_rows, limit=12)
    techniques = [name for name, _ in ranked]
    index = {
        (row["alias"], row["technique_name"]): float(row.get("mean_overall_score", 0.0))
        for row in technique_rows
        if row["technique_name"] in techniques
    }
    matrix = [[index.get((alias, technique), 0.0) for technique in techniques] for alias in aliases]

    cmap = LinearSegmentedColormap.from_list("technique_heat", ["#f5efe4", "#9ed8db", "#467599", "#1d3557"])
    fig, ax = plt.subplots(figsize=(max(12.5, 0.9 * len(techniques)), max(5.6, 0.58 * len(aliases))))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=max(0.35, max(max(row) for row in matrix)))
    ax.set_xticks(range(len(techniques)))
    ax.set_xticklabels([_wrap_label(name, 18) for name in techniques], rotation=0, ha="center")
    ax.set_yticks(range(len(aliases)))
    ax.set_yticklabels(aliases)
    ax.set_title("Technique Landscape")
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            text = ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white")
            text.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="#00000055")])
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Mean StrongREJECT overall score")
    _save_figure(fig, output_path)


def _plot_scaling(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    rows = [row for row in summary_rows if row.get("params_b") not in (None, "", 0)]
    if not rows:
        return

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("family", "unknown"))].append(row)

    fig, ax = plt.subplots(figsize=(10.4, 6.7))
    for family, family_rows in sorted(by_family.items()):
        family_rows.sort(key=lambda row: float(row["params_b"]))
        xs = [math.log10(float(row["params_b"])) for row in family_rows]
        ys = [float(row.get("sr_mean_overall", row.get("risk_score", 0.0))) for row in family_rows]
        color = _family_color(family)
        ax.plot(xs, ys, marker="o", linewidth=2.6, color=color, label=family)
        for x_value, y_value, row in zip(xs, ys, family_rows):
            text = ax.text(x_value, y_value + 0.004, row["alias"], fontsize=8.3, ha="center")
            _outline_text(text)
        if len(xs) >= 2:
            slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
            ax.text(xs[-1], ys[-1] - 0.014, f"slope {slope:+.3f}", fontsize=8, ha="right", color=color)

    ax.set_xlabel(r"$\log_{10}(\mathrm{parameters\ in\ billions})$")
    ax.set_ylabel("StrongREJECT overall score")
    if len(by_family) == 1:
        family = next(iter(by_family))
        ax.set_title(f"Exploratory Size Trend on Reachable {family.title()} Models")
    else:
        ax.set_title("Exploratory Size Trend on Reachable Size-Annotated Families")
    ax.yaxis.grid(True, color=GRID)
    ax.xaxis.grid(True, color=GRID)
    ax.legend(loc="best")
    _save_figure(fig, output_path)


def _draw_provider_status(ax: Any, status_rows: list[dict[str, Any]], *, title: str) -> None:
    if not status_rows:
        ax.axis("off")
        return
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in status_rows:
        provider = _status_provider(row)
        counts[provider]["ok" if row.get("ok") else "blocked"] += 1

    providers = sorted(counts)
    ok_vals = [counts[p]["ok"] for p in providers]
    blocked_vals = [counts[p]["blocked"] for p in providers]

    ax.barh(providers, blocked_vals, color=BLOCKED, label="blocked", alpha=0.88)
    ax.barh(providers, ok_vals, left=blocked_vals, color=OK, label="reachable", alpha=0.92)
    ax.set_title(title)
    ax.set_xlabel("Registered aliases")
    ax.xaxis.grid(True, color=GRID)
    for idx, provider in enumerate(providers):
        total = ok_vals[idx] + blocked_vals[idx]
        text = ax.text(total + 0.08, idx, f"{ok_vals[idx]}/{total} reachable", va="center", fontsize=8.5)
        _outline_text(text)
    ax.legend(loc="lower right")


def _plot_provider_status(status_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not status_rows:
        return
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    _draw_provider_status(ax, status_rows, title="Registry Reachability by Provider")
    _save_figure(fig, output_path)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    if not fig.get_constrained_layout():
        fig.tight_layout()
    base = output_path.with_suffix("")
    for suffix in FIGURE_FORMATS:
        target = base.with_suffix(suffix)
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if suffix == ".png":
            save_kwargs["dpi"] = 360
        fig.savefig(target, **save_kwargs)
    plt.close(fig)


def _write_model_summary_table(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Model & Family & Risk & Hit rate & Error rate & SR overall & Judge comply \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        hit_rate = f"{float(row.get('success_rate', 0.0) or 0.0):.0%}".replace("%", "\\%")
        error_rate = f"{float(row.get('error_rate', 0.0) or 0.0):.0%}".replace("%", "\\%")
        judge_comply = f"{(1.0 - float(row.get('sr_refusal_rate', 0.0) or 0.0)):.0%}".replace("%", "\\%")
        lines.append(
            f"{_latex_escape(row['alias'])} & "
            f"{_latex_escape(str(row.get('family', '-')))} & "
            f"{float(row.get('risk_score', 0.0)):.2f} & "
            f"{hit_rate} & "
            f"{error_rate} & "
            f"{float(row.get('sr_mean_overall', 0.0) or 0.0):.3f} & "
            f"{judge_comply} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output_path.write_text("\n".join(lines))


def _write_top_techniques_table(technique_rows: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Technique & Mean SR overall \\\\",
        "\\midrule",
    ]
    for name, score in _top_techniques(technique_rows, limit=10):
        lines.append(f"{_latex_escape(name)} & {score:.3f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output_path.write_text("\n".join(lines))


def _write_blockers_table(status_rows: list[dict[str, Any]], output_path: Path) -> None:
    blocked_rows = [row for row in status_rows if not row.get("ok")]
    if not blocked_rows:
        output_path.write_text("% No blockers recorded.\n")
        return

    lines = [
        "\\begin{longtable}{>{\\raggedright\\arraybackslash}p{0.22\\textwidth}>{\\raggedright\\arraybackslash}p{0.7\\textwidth}}",
        "\\toprule",
        "Alias & Blocking condition \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for row in blocked_rows:
        detail = str(row.get("error", "")).replace("\n", " ")
        detail = textwrap.shorten(detail, width=180, placeholder="...")
        lines.append(f"{_latex_escape(str(row.get('spec', '-')))} & {_latex_escape(detail)} \\\\")
    lines.extend(["\\bottomrule", "\\end{longtable}", ""])
    output_path.write_text("\n".join(lines))


def _write_markdown_report(
    payload: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    technique_rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    output_path: Path,
    artifacts: dict[str, Path],
) -> None:
    blocked = [row for row in status_rows if not row.get("ok")]
    top_risk = summary_rows[:5]
    top_sr = sorted(summary_rows, key=lambda row: float(row.get("sr_mean_overall", 0.0) or 0.0), reverse=True)[:5]
    top_techniques = _top_techniques(technique_rows, limit=10)
    scaling_note = _scaling_note(summary_rows)

    lines: list[str] = []
    lines.append("# Publication Report")
    lines.append("")
    lines.append(f"- Campaign: `{payload.get('campaign', 'unknown')}`")
    lines.append(f"- Evaluator: `{payload.get('evaluator', 'none')}`")
    lines.append(f"- Judge: `{payload.get('judge') or 'default'}`")
    lines.append(f"- Reachable models completed: {len(summary_rows)}")
    lines.append(f"- Registry aliases blocked in preflight: {len(blocked)}")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    if top_risk:
        lines.append(
            f"- Highest risk score: `{top_risk[0]['alias']}` at `{float(top_risk[0].get('risk_score', 0.0)):.2f}`."
        )
    if top_sr:
        lines.append(
            f"- Highest StrongREJECT overall score: `{top_sr[0]['alias']}` at `{float(top_sr[0].get('sr_mean_overall', 0.0) or 0.0):.3f}`."
        )
    if top_techniques:
        lines.append(
            f"- Most effective technique: `{top_techniques[0][0]}` with mean score `{top_techniques[0][1]:.3f}`."
        )
    if summary_rows:
        highest_error = max(summary_rows, key=lambda row: float(row.get("error_rate", 0.0) or 0.0))
        lines.append(
            f"- Highest probe error rate: `{highest_error['alias']}` at `{float(highest_error.get('error_rate', 0.0) or 0.0):.0%}`."
        )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for key in (
        "overview_dashboard_pdf",
        "risk_bar_pdf",
        "frontier_pdf",
        "metric_heatmap_pdf",
        "technique_heatmap_pdf",
        "scaling_plot_pdf",
        "provider_status_pdf",
    ):
        if key in artifacts:
            rel = artifacts[key].relative_to(output_path.parent)
            png = rel.with_suffix(".png")
            lines.append(f"### {key.replace('_pdf', '').replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"![]({png.as_posix()})")
            lines.append("")
    lines.append("## Top Models")
    lines.append("")
    lines.append("| Model | Family | Risk | Hit rate | Error rate | SR overall | Judge comply |")
    lines.append("|-------|--------|------|----------|------------|------------|--------------|")
    for row in summary_rows:
        lines.append(
            f"| {row['alias']} | {row['family']} | {float(row.get('risk_score', 0.0)):.2f} | "
            f"{float(row.get('success_rate', 0.0) or 0.0):.0%} | "
            f"{float(row.get('error_rate', 0.0) or 0.0):.0%} | "
            f"{float(row.get('sr_mean_overall', 0.0) or 0.0):.3f} | "
            f"{(1.0 - float(row.get('sr_refusal_rate', 0.0) or 0.0)):.0%} |"
        )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(f"- {scaling_note}")
    lines.append(
        f"- Registry preflight blocked {len(blocked)} aliases, so the empirical section reflects the reachable subset rather than the full configured registry."
    )
    lines.append("")
    lines.append("## Strongest Techniques")
    lines.append("")
    lines.append("| Technique | Mean SR overall |")
    lines.append("|-----------|-----------------|")
    for name, score in top_techniques:
        lines.append(f"| {name} | {score:.3f} |")
    lines.append("")
    lines.append("## LaTeX Assets")
    lines.append("")
    for key in ("report_tex", "model_summary_tex", "top_techniques_tex", "execution_blockers_tex"):
        if key in artifacts:
            rel = artifacts[key].relative_to(output_path.parent)
            lines.append(f"- `{key}`: `{rel}`")
    lines.append("")
    lines.append("## Authorship And Tooling Disclosure")
    lines.append("")
    lines.append(DISCLOSURE_TEXT)
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n")


def _write_tex_report(
    payload: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    technique_rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    top_risk = summary_rows[:3]
    top_sr = sorted(summary_rows, key=lambda row: float(row.get("sr_mean_overall", 0.0) or 0.0), reverse=True)[:3]
    top_techniques = _top_techniques(technique_rows, limit=5)
    blocked = [row for row in status_rows if not row.get("ok")]
    scaling_note = _scaling_note(summary_rows)

    lines: list[str] = [
        "\\documentclass[11pt]{article}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage{lmodern}",
        "\\usepackage{microtype}",
        "\\usepackage{booktabs}",
        "\\usepackage{longtable}",
        "\\usepackage{graphicx}",
        "\\usepackage{float}",
        "\\usepackage{hyperref}",
        "\\usepackage{xcolor}",
        "\\usepackage{array}",
        "\\usepackage{caption}",
        "\\usepackage{subcaption}",
        "\\usepackage{siunitx}",
        "\\usepackage{enumitem}",
        "\\sisetup{detect-all=true}",
        "\\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}",
        "\\title{Red-Agent Vulnerability Assessment}",
        "\\author{Automated Publication Bundle}",
        "\\date{\\today}",
        "\\begin{document}",
        "\\maketitle",
        "\\begin{abstract}",
        "This report summarizes a full adversarial scan over the currently reachable model subset, "
        "combining campaign hit rates with StrongREJECT-style scoring and an explicit registry-wide "
        "provider reachability preflight. The figure set is exported as PDF, SVG, and PNG for direct "
        "paper integration.",
        "\\end{abstract}",
        "\\section{Executive Summary}",
        "\\begin{itemize}[leftmargin=1.2em]",
    ]

    if top_risk:
        lines.append(
            f"\\item Highest measured aggregate risk: \\textbf{{{_latex_escape(top_risk[0]['alias'])}}} "
            f"with risk score \\textbf{{{float(top_risk[0].get('risk_score', 0.0)):.2f}}}."
        )
    if top_sr:
        lines.append(
            f"\\item Highest StrongREJECT overall score: \\textbf{{{_latex_escape(top_sr[0]['alias'])}}} "
            f"at \\textbf{{{float(top_sr[0].get('sr_mean_overall', 0.0) or 0.0):.3f}}}."
        )
    if top_techniques:
        lines.append(
            f"\\item Most effective technique across completed runs: \\textbf{{{_latex_escape(top_techniques[0][0])}}} "
            f"with mean score \\textbf{{{top_techniques[0][1]:.3f}}}."
        )
    if summary_rows:
        highest_error = max(summary_rows, key=lambda row: float(row.get("error_rate", 0.0) or 0.0))
        highest_error_rate = f"{float(highest_error.get('error_rate', 0.0) or 0.0):.0%}".replace("%", "\\%")
        lines.append(
            f"\\item Highest probe error rate: \\textbf{{{_latex_escape(highest_error['alias'])}}} "
            f"at \\textbf{{{highest_error_rate}}}."
        )
    lines.append(
        f"\\item Registry preflight blocked \\textbf{{{len(blocked)}}} configured aliases, preventing those rows from being treated as true negatives."
    )
    lines.extend(
        [
            "\\end{itemize}",
            "\\begin{figure}[H]",
            "\\centering",
            "\\includegraphics[width=\\textwidth]{plots/overview_dashboard.pdf}",
            "\\caption{Executive dashboard: risk ranking, attack frontier, top techniques, and registry reachability.}",
            "\\end{figure}",
            "\\section{Model-Level Results}",
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{Model summary on the reachable subset.}",
            "\\input{tables/model_summary.tex}",
            "\\end{table}",
            "\\begin{figure}[H]",
            "\\centering",
            "\\includegraphics[width=0.84\\textwidth]{plots/frontier.pdf}",
            "\\caption{Attack frontier. Bubble area increases with aggregate risk score.}",
            "\\end{figure}",
            "\\begin{figure}[H]",
            "\\centering",
            "\\includegraphics[width=0.92\\textwidth]{plots/metric_heatmap.pdf}",
            "\\caption{Normalized heatmap where darker cells indicate greater vulnerability intensity.}",
            "\\end{figure}",
            "\\section{Technique Landscape}",
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{Top techniques by mean StrongREJECT overall score.}",
            "\\input{tables/top_techniques.tex}",
            "\\end{table}",
            "\\begin{figure}[H]",
            "\\centering",
            "\\includegraphics[width=\\textwidth]{plots/technique_heatmap.pdf}",
            "\\caption{Per-model technique landscape over the highest-scoring techniques.}",
            "\\end{figure}",
            "\\section{Scaling and Reachability}",
            "\\begin{figure}[H]",
            "\\centering",
            "\\includegraphics[width=0.84\\textwidth]{plots/scaling_plot.pdf}",
            f"\\caption{{{_latex_escape(scaling_note)}}}",
            "\\end{figure}",
            "\\begin{figure}[H]",
            "\\centering",
            "\\includegraphics[width=0.84\\textwidth]{plots/provider_status.pdf}",
            "\\caption{Registry reachability by provider.}",
            "\\end{figure}",
            "\\section{Limitations}",
            _latex_escape(scaling_note),
            "",
            f"The reachable subset comprised {len(summary_rows)} completed models; blocked providers accounted for {len(blocked)} configured aliases.",
            "\\section{Execution Blockers}",
            "The following aliases were blocked by quota, invalid credentials, or stale model identifiers during the registry preflight.",
            "\\input{tables/execution_blockers.tex}",
            "\\section{Reproducibility}",
            f"The campaign type was \\emph{{{_latex_escape(str(payload.get('campaign', 'unknown')))}}}; "
            f"the evaluator was \\emph{{{_latex_escape(str(payload.get('evaluator', 'none')))}}}; "
            f"the judge model was {_latex_escape(str(payload.get('judge') or 'default'))}.",
            "\\section{Authorship and Tooling Disclosure}",
            _latex_escape(DISCLOSURE_TEXT),
            "\\end{document}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    escaped = text
    for source, target in replacements.items():
        escaped = escaped.replace(source, target)
    return escaped


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-grade report assets from comparison JSON.")
    parser.add_argument("--input", required=True, help="Path to comparison JSON output.")
    parser.add_argument("--output-dir", required=True, help="Directory for report artifacts.")
    parser.add_argument("--status-json", default=None, help="Optional path to provider preflight status JSON.")
    args = parser.parse_args()
    build_publication_report(args.input, args.output_dir, status_json=args.status_json)


if __name__ == "__main__":
    main()
