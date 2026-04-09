# Paper Notes

## Directly Measured

- The curated comparison run executed `20` probes per model across `11` models.
- Providers represented in the curated registry: OpenAI, Anthropic, Google, and Mistral.
- Highest aggregate risk score: `gemini-2.5-pro` at `7.60`.
- Highest campaign hit rate: `mistral-small` at `0.75`.
- Highest operational error rate: `gemini-2.5-pro` at `0.45`.
- Preflight reachability for the curated run is `11/11` with no blocked aliases.
- Qualitative traces still show override/persona-hijack behaviors in Mistral trajectories.

## Inferred Or Synthesized

- Curating the registry to fully testable aliases improves comparability and reduces
  confounding from credential/quota failures.
- Risk, hit rate, and error rate diverge by family; no single scalar captures the
  complete operational picture.
- The engine package remains a systems contribution; empirical results in this paper
  still come from the stable comparison-runner path.

## Remaining Weaknesses

- Registry coverage is intentionally narrower (11 aliases), so claims do not extend to
  excluded providers/families.
- Evaluator-judge metrics were not the focus of this refreshed run.
- The ATT&CK-inspired probe set differs from external benchmark prompt sets.
- Size-trend interpretation remains exploratory due to uneven family coverage.
