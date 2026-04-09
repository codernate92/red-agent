"""Generate comparative visuals across campaign metrics and judge metrics.

Inputs:
- Enriched comparison JSON (with SR fields in summary rows)
- Optional ClearHarm comparison JSON for benchmark overlay
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def _load_summary(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = payload.get("summary", [])
    rows = [r for r in rows if r.get("alias") and r.get("error") in (None, "")]
    return rows


def _to_float(v, default=0.0):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Build comparative benchmark figure")
    parser.add_argument("--input", required=True, help="Enriched comparison JSON")
    parser.add_argument("--clearharm", default=None, help="Optional clearharm comparison JSON")
    parser.add_argument("--output", required=True, help="Output image path (.png/.pdf/.svg)")
    args = parser.parse_args()

    rows = _load_summary(Path(args.input))
    rows.sort(key=lambda r: _to_float(r.get("risk_score")), reverse=True)

    aliases = [r["alias"] for r in rows]
    risk = np.array([_to_float(r.get("risk_score")) for r in rows])
    hit = np.array([_to_float(r.get("success_rate")) for r in rows])
    err = np.array([_to_float(r.get("error_rate")) for r in rows])
    sr = np.array([_to_float(r.get("sr_mean_overall")) for r in rows])

    clearharm_hit = None
    if args.clearharm:
        clear_rows = {r["alias"]: r for r in _load_summary(Path(args.clearharm))}
        clearharm_hit = np.array([_to_float(clear_rows.get(a, {}).get("success_rate"), np.nan) for a in aliases])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "figure.facecolor": "#F8FAFC",
            "axes.facecolor": "#FCFDFE",
        }
    )
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # Panel 1: frontier scatter
    ax1 = fig.add_subplot(gs[0, 0])
    sizes = 120 + err * 700
    sc = ax1.scatter(hit, risk, s=sizes, c=sr, cmap="viridis", alpha=0.9, edgecolor="black", linewidth=0.5)
    for i, a in enumerate(aliases):
        ax1.annotate(a, (hit[i], risk[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax1.set_title("Risk vs Hit Rate (bubble size = error, color = SR overall)")
    ax1.set_xlabel("Campaign hit rate")
    ax1.set_ylabel("Aggregate risk score")
    ax1.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax1.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax1, shrink=0.86)
    cbar.set_label("SR mean overall")

    # Panel 2: grouped bars baseline vs clearharm (if available)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(aliases))
    w = 0.38
    ax2.bar(x - w / 2, hit, width=w, label="Campaign hit rate", color="#1F7A8C")
    if clearharm_hit is not None:
        ax2.bar(x + w / 2, clearharm_hit, width=w, label="ClearHarm hit rate", color="#E09F3E")
    ax2.set_xticks(x)
    ax2.set_xticklabels(aliases, rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Baseline vs ClearHarm Hit Rate")
    ax2.set_ylabel("Rate")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(frameon=False, fontsize=9)

    # Panel 3: heatmap of normalized metrics
    ax3 = fig.add_subplot(gs[1, :])
    metric_names = ["risk", "hit", "error", "sr"]
    M = np.vstack([risk, hit, err, sr]).T
    denom = np.maximum(M.max(axis=0) - M.min(axis=0), 1e-9)
    Mn = (M - M.min(axis=0)) / denom
    im = ax3.imshow(Mn, aspect="auto", cmap="YlGnBu")
    ax3.set_xticks(np.arange(len(metric_names)))
    ax3.set_xticklabels(metric_names)
    ax3.set_yticks(np.arange(len(aliases)))
    ax3.set_yticklabels(aliases, fontsize=8)
    ax3.set_title("Normalized Metric Profile by Model")
    for i in range(len(aliases)):
        for j in range(len(metric_names)):
            ax3.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="#111111", fontsize=7)
    fig.colorbar(im, ax=ax3, shrink=0.6, label="normalized value")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
