# red-agent

Authorized red-teaming framework for LLMs and agentic systems. It runs structured adversarial campaigns, scores findings with CVSS-like severity and StrongREJECT-style evaluator outputs, and compares behavior across providers and model scales.

## What it does

- Runs ATT&CK-inspired prompt-injection, exfiltration, privilege-escalation, persistence, goal-hijacking, and defense-evasion probes.
- Supports real model backends through OpenAI, Anthropic, Google Gemini, Mistral, Together, Ollama, and vLLM.
- Logs model interactions to JSONL for auditability.
- Produces JSON and Markdown reports with vulnerability scoring, attack-surface summaries, and optional evaluator aggregates.
- Compares model families using the shared registry in [`core/model_registry.py`](/Users/nathanheath/red-agent/core/model_registry.py).

## Layout

```text
red-agent/
  analysis/    evaluator logic, attack-surface maps, reports, scoring
  attacks/     adversarial probe suites
  core/        taxonomy, targets, registry, probes, campaigns, trajectories
  harness/     campaign and comparison runners
  tests/       pytest coverage for core behavior
  cli.py       command-line entrypoint
```

## Installed Capabilities

### Targets

- `openai`
- `anthropic`
- `google`
- `mistral`
- `together`
- `ollama`
- `vllm`

All provider wrappers inherit from [`core/targets/base_target.py`](/Users/nathanheath/red-agent/core/targets/base_target.py) and share retry/backoff, optional request-rate limiting, timeout handling, and JSONL trajectory logging.

### Evaluators

- `strongreject`
- `sentiment`
- `code_backdoor`

Evaluator interfaces live under [`analysis/evaluators/`](/Users/nathanheath/red-agent/analysis/evaluators) and can be attached to both `scan` and `compare`.

### Registry

The current registry includes:

- OpenAI closed: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`
- Anthropic: `claude-sonnet-4`, `claude-sonnet-4-6`, `claude-sonnet-4-5`, `claude-haiku-4-5`, `claude-opus-4-6`
- Google: `gemini-2.5-pro`
- Mistral family: `mistral-small`, `mistral-large`

Use `python cli.py targets` to see the active list plus API-key status.

## Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Create a local `.env` from [`.env.example`](/Users/nathanheath/red-agent/.env.example). `cli.py` loads it automatically with `python-dotenv` before target construction.

Relevant variables:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `MISTRAL_API_KEY`
- Optional: `TOGETHER_API_KEY` (for raw `together:<model>` specs outside the curated registry)
- Optional: `VLLM_BASE_URL`, `OLLAMA_BASE_URL`

## CLI

List targets:

```bash
python cli.py targets
```

Single-model scan:

```bash
python cli.py scan --target gpt-4o-mini --campaign quick
python cli.py scan --target claude-sonnet-4-6 --campaign full --trajectory-log trajectories/claude-sonnet-4-6.jsonl
python cli.py scan --target mistral-large --campaign quick --evaluate strongreject
```

Cross-model comparison:

```bash
python cli.py compare --family claude --campaign quick
python cli.py compare --targets gpt-4o,gpt-4o-mini,gpt-4-turbo,gemini-2.5-pro,mistral-small,mistral-large --campaign quick --evaluate strongreject --reports-dir reports/
```

Existing-result analysis:

```bash
python cli.py analyze --results reports/example.json --format markdown
```

Offline demo:

```bash
python cli.py demo --target realistic
```

## Reporting

`scan` and `compare` can emit:

- JSON campaign or comparison output
- Markdown reports per model
- StrongREJECT-style aggregates when `--evaluate` is enabled
- JSONL trajectory logs for every prompt/response exchange

The report generator is [`analysis/report.py`](/Users/nathanheath/red-agent/analysis/report.py).

## Testing

Run the suite with:

```bash
python3 -m pytest -q
```

The tests cover campaign execution, taxonomy behavior, reporting, evaluator aggregation, and registry resolution.

## Intended Use

This repository is for authorized security research and defensive evaluation only. Use it to assess systems you own, operate, or are explicitly permitted to test.

## License

MIT
