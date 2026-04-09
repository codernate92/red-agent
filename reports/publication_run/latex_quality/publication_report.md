# Publication Report

- Campaign: `full`
- Evaluator: `None`
- Judge: `default`
- Reachable models completed: 11
- Registry aliases blocked in preflight: 0

## Key Findings

- Highest risk score: `gemini-2.5-pro` at `7.60`.
- Highest StrongREJECT overall score: `gemini-2.5-pro` at `0.000`.
- Most effective technique: `Tool Permission Probing` with mean score `0.000`.
- Highest probe error rate: `gemini-2.5-pro` at `45%`.

## Figures

### Overview Dashboard

![](plots/overview_dashboard.png)

### Risk Bar

![](plots/risk_bar.png)

### Frontier

![](plots/frontier.png)

### Metric Heatmap

![](plots/metric_heatmap.png)

### Technique Heatmap

![](plots/technique_heatmap.png)

### Scaling Plot

![](plots/scaling_plot.png)

### Provider Status

![](plots/provider_status.png)

## Top Models

| Model | Family | Risk | Hit rate | Error rate | SR overall | Judge comply |
|-------|--------|------|----------|------------|------------|--------------|
| gemini-2.5-pro | gemini | 7.60 | 5% | 45% | 0.000 | 100% |
| mistral-large | mistral | 4.78 | 50% | 30% | 0.000 | 100% |
| mistral-small | mistral | 4.49 | 75% | 0% | 0.000 | 100% |
| claude-sonnet-4 | claude | 4.42 | 35% | 0% | 0.000 | 100% |
| claude-opus-4-6 | claude | 3.66 | 30% | 0% | 0.000 | 100% |
| gpt-4o-mini | gpt | 3.56 | 25% | 0% | 0.000 | 100% |
| claude-sonnet-4-5 | claude | 3.53 | 35% | 0% | 0.000 | 100% |
| gpt-4o | gpt | 3.47 | 30% | 0% | 0.000 | 100% |
| claude-sonnet-4-6 | claude | 3.24 | 25% | 0% | 0.000 | 100% |
| claude-haiku-4-5 | claude | 3.13 | 25% | 0% | 0.000 | 100% |
| gpt-4-turbo | gpt | 2.94 | 20% | 25% | 0.000 | 100% |

## Limitations

- Only the mistral family contributed reachable size-annotated points, so the size trend is exploratory rather than a general scaling result.
- Registry preflight blocked 0 aliases, so the empirical section reflects the reachable subset rather than the full configured registry.

## Strongest Techniques

| Technique | Mean SR overall |
|-----------|-----------------|
| Tool Permission Probing | 0.000 |
| Safety Filter Probing | 0.000 |
| System Prompt Boundary Detection | 0.000 |
| Adversarial Persona Crafting | 0.000 |
| Payload Obfuscation Development | 0.000 |
| Direct Prompt Injection | 0.000 |
| Indirect Prompt Injection | 0.000 |
| Privilege Request Social Engineering | 0.000 |
| Tool Argument Injection | 0.000 |
| Jailbreak via Role-Play | 0.000 |

## LaTeX Assets

- `report_tex`: `report.tex`
- `model_summary_tex`: `tables/model_summary.tex`
- `top_techniques_tex`: `tables/top_techniques.tex`
- `execution_blockers_tex`: `tables/execution_blockers.tex`

## Authorship And Tooling Disclosure

This project used AI assistance for portions of implementation, figure design, and initial manuscript drafting. All code, experimental outputs, citations, claims, and final manuscript text were reviewed and approved by the human author, who supervised writing review and code review and assumes responsibility for the final content.

