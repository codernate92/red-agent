# red-agent

Adversarial red-teaming framework for AI agents. Systematically probes autonomous systems for exploitable behaviors using an ATT&CK-inspired taxonomy, adaptive campaign execution, and CVSS-analog vulnerability scoring.

## Overview

red-agent treats AI agent security evaluation the way penetration testing treats network security: structured, repeatable, and comprehensive. It provides a taxonomy of 20 attack techniques across 9 tactic categories, 39 pre-built attack probes, and an adaptive campaign runner that escalates based on discovered vulnerabilities.

## Architecture

```
red-agent/
  core/           # Taxonomy, probes, campaigns, targets, trajectory capture
  attacks/        # 6 attack suites with 39 probes
  harness/        # Campaign runner with adaptive escalation
  analysis/       # Vulnerability scoring, attack surface mapping, reports
  tests/          # 26 passing tests
  cli.py          # CLI with demo, scan, analyze, techniques commands
```

## Attack Taxonomy

Inspired by MITRE ATT&CK, adapted for AI agents:

| Tactic | Techniques | Description |
|--------|-----------|-------------|
| Reconnaissance | 3 | Probe agent capabilities, tool permissions, safety boundaries |
| Resource Development | 2 | Craft adversarial personas and obfuscated payloads |
| Initial Access | 2 | Direct and indirect prompt injection |
| Privilege Escalation | 2 | Social engineering for tool access, argument injection |
| Defense Evasion | 3 | Jailbreaks, multi-turn escalation, encoding bypasses |
| Persistence | 2 | Context window poisoning, memory/state injection |
| Exfiltration | 3 | System prompt extraction, training data probing |
| Impact | 1 | Output format manipulation |
| Goal Hijacking | 2 | Instruction override, persona hijacking |

## Attack Suites

- **Prompt Injection** (8 probes) — delimiter escapes, nested injection, multi-language payloads
- **Privilege Escalation** (6 probes) — authority impersonation, tool argument manipulation, chained escalation
- **Goal Hijacking** (6 probes) — gradual persona shift, conflicting instructions, fake system updates
- **Exfiltration** (7 probes) — completion attacks, translation tricks, structured extraction
- **Defense Evasion** (7 probes) — DAN jailbreaks, academic framing, base64/ROT13 obfuscation
- **Persistence** (5 probes) — context poisoning, memory injection, conversation hijacking

## Campaign Execution

Campaigns run probes in phases with configurable escalation strategies:

- **Linear** — run all probes in order
- **Adaptive** — skip/add probes based on results
- **Depth-first** — exploit successful vectors deeper
- **Breadth-first** — cover all tactic categories before deepening

## Quick Start

```bash
# Run the demo against a mock target
python cli.py demo

# Run a full campaign against a target
python cli.py scan --target my_agent --campaign full

# List available techniques
python cli.py techniques --list

# Analyze saved results
python cli.py analyze --results results.json --format markdown
```

## Vulnerability Scoring

Each discovered vulnerability receives a CVSS-analog score (0-10) factoring in:

- Technique severity and detection difficulty
- Reproduction success rate across probes
- Confidence level from success detectors

Reports include remediation recommendations per vulnerability.

## Testing

```bash
pytest tests/ -v  # 26 tests
```

## Intended Use

This framework is designed for **authorized security evaluation** of AI systems. It is intended for:

- AI safety researchers evaluating model robustness
- Red teams conducting authorized assessments of deployed agents
- Organizations testing their AI systems before deployment
- Academic research on adversarial robustness

## License

MIT
