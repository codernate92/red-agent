# Paper Notes

## Directly Measured

- The archived comparison run executed `20` probes per model across `7` models:
  five Anthropic aliases and two Mistral aliases.
- The executed run used the `full` campaign and `strongreject` evaluation with
  judge `anthropic:claude-sonnet-4-6`.
- Highest aggregate risk score: `claude-sonnet-4` at `4.3917`.
- Highest `StrongREJECT` overall score: `mistral-large` at `0.11875`.
- Highest operational error rate: `mistral-large` at `0.35`.
- Highest mean technique score: `Tool Permission Probing` at `0.330`.
- The combined reachability table records `13` blocked aliases:
  `8` invalid Together API keys, `4` quota-related failures, and `1` stale
  Gemini preview model identifier.
- Concrete qualitative traces show both `mistral-small` and `mistral-large`
  responding with `COMPROMISED` and adopting the `ShadowGPT` persona.

## Inferred Or Synthesized

- The provider reachability table in the paper treats the `7` executed models as
  successful registry entries and combines them with subsequent preflight
  failures for the remaining `13` aliases.
- The paper describes the new `engine/` package as a systems contribution, but
  the archived empirical section is based on the stable comparison-runner path
  rather than an end-to-end live engine sweep.
- The qualitative interpretation that Mistral models displayed stronger
  evaluator-rated jailbreak behavior than the Claude cluster is based on the
  archived `StrongREJECT` scores and trace examples, not on human re-labeling.

## Remaining Weaknesses

- Only `7/20` configured aliases were executed successfully, so the empirical
  sample is narrow.
- The run covers only the Claude and Mistral families in practice.
- `StrongREJECT` scores were produced by a single judge model/provider rather
  than by multiple judges or human annotation.
- The ATT&CK-inspired probe set is not the same as the public HarmBench or
  `StrongREJECT` harmful-prompt benchmarks, so benchmark-level comparisons
  should be treated cautiously.
- The exploratory size trend is based only on the reachable Mistral family and
  should not be read as a general scaling law.
