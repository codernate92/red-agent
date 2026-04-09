# Paper Build

This directory contains an arXiv-style manuscript bundle built from the archived
`red-agent` experiment artifacts.

## Build

From the repository root:

```bash
cd paper
make pdf
```

This writes the compiled PDF to `paper/main.pdf`.

## ArXiv Source Bundle

Generate a clean source upload bundle from the same paper directory:

```bash
make arxiv
```

This creates:

- `paper/dist/red-agent-arxiv-source.tar.gz`

The archive contains `main.tex`, `abstract.tex`, `references.bib`, `sections/`,
`tables/`, `figures/`, and a generated `main.bbl` when available.

## Inputs

The paper is grounded in the existing artifacts under:

- `reports/publication_run/comparison.json`
- `reports/publication_run/provider_status.json`
- `reports/publication_run/latex_quality/`
- `reports/publication_run/per_model/`
- `trajectories/publication_run/`

The result figures copied into `paper/figures/` came from the verified
publication-report pipeline under `reports/publication_run/latex_quality/plots/`.

## Re-generating Result Figures

If the underlying experiment artifacts change, regenerate the publication bundle
first:

```bash
python3 -m analysis.publication_report \
  --input reports/publication_run/comparison.json \
  --status-json reports/publication_run/provider_status.json \
  --output-dir reports/publication_run/latex_quality
```

Then update the copied PDFs in `paper/figures/` before rebuilding the paper
and regenerating the ArXiv bundle.

## Files

- `main.tex`: top-level manuscript
- `abstract.tex`: abstract text
- `sections/`: manuscript sections
- `figures/`: copied result plots plus TikZ figure snippets
- `tables/`: manuscript tables and appendix tables
- `references.bib`: verified citations
- `NOTES.md`: measured facts, inferred statements, and known weaknesses
