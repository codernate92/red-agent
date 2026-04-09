# red-agent Report Template

This file is a reusable report skeleton for real experiment runs. It does not contain fabricated findings.

## Objective

Document a reproducible red-team sweep across one or more registered models using:

- campaign type: `quick` or `full`
- evaluator: `strongreject`, `sentiment`, or `code_backdoor`
- registry targets from [`core/model_registry.py`](/Users/nathanheath/red-agent/core/model_registry.py)

## Run Metadata

- Date:
- Operator:
- Git revision:
- Command(s) executed:
- Judge model:
- Reports directory:
- Trajectory directory:

## Models Tested

- Model aliases:
- Families:
- Provider mix:

## Methodology

1. Load provider credentials from `.env`.
2. Run `python cli.py targets` to confirm registry visibility and key presence.
3. Execute either:
   - `python cli.py scan ...` for a single target
   - `python cli.py compare ...` for a family or explicit alias list
4. Save JSON, Markdown, and JSONL artifacts.
5. Summarize risk scores, evaluator aggregates, and notable vulnerabilities.

## Artifacts

- Per-model JSON reports:
- Per-model Markdown reports:
- JSONL trajectories:
- Comparison JSON:

## Summary

### High-Level Findings

- 

### StrongREJECT / Evaluator Summary

- Mean refusal rate:
- Mean overall score:
- Highest-severity model:

### Cross-Model Notes

- Universal findings:
- Model-specific findings:
- Scaling observations:

## Detailed Findings

### Critical

- 

### High

- 

### Medium / Low

- 

## Remediation Themes

- Prompt-boundary hardening
- Tool authorization and least privilege
- Context and memory isolation
- Output-side leak prevention

## Follow-Ups

- Run deeper campaign:
- Expand prompt set:
- Re-test after mitigation:
