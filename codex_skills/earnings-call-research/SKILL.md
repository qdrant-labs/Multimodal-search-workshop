---
name: earnings-call-research
description: Build evidence-backed research briefs from open-ended business, markets, data-science, or journalism prompts using the Multimodal-search-workshop earnings-call corpus. Use when asked to investigate a hypothesis, find transcript evidence, compare company language, surface supporting/counter evidence, or turn broad research questions into cited earnings-call findings.
---

# Earnings Call Research

Use this skill to turn a broad research prompt into a finished earnings-call analysis without modifying the workshop exercises. The bundled script prints progress on stderr, prints the final analysis on stdout, and writes an evidence package with a package `README.md`, transcript quotes, point IDs, audio clip paths, and live AskNews context when `ASKNEWS_API_KEY` is available.

## Quick Start

From the `Multimodal-search-workshop` repo root, run:

```bash
.venv/bin/python codex_skills/earnings-call-research/scripts/research_brief.py \
  "research cyclic investment in and by AI companies and find evidence for it in earnings calls"
```

Use transcript-only mode when AskNews credentials or network are unavailable:

```bash
.venv/bin/python codex_skills/earnings-call-research/scripts/research_brief.py \
  --no-asknews \
  "find evidence for AI capex cyclicality"
```

For structured output:

```bash
.venv/bin/python codex_skills/earnings-call-research/scripts/research_brief.py \
  --json \
  "compare AI investment language across NVDA, AMZN, AAPL, and TSLA" > /tmp/earnings-brief.json
```

## Workflow

1. Run the CLI with the user's prompt as written.
2. Give the user the finished analysis from stdout, not the raw evidence table.
3. Mention the `research_outputs/...` evidence package path.
4. Check the package `README.md` first, then `asknews_context.md` for live external validation and contradictions.
5. Keep only claims supported by direct or contextual evidence.
6. Use point IDs with `get_audio_clip` before treating a quote as publishable.
7. Search for counter-evidence by adding terms such as `risk`, `slowdown`, `digestion`, `ROI`, `capacity`, or `demand normalization`.

## Evidence Discipline

Read `references/evidence-standards.md` when producing a report for publication, external review, or sensitive business analysis.

Never turn retrieval rank into proof. State whether a quote is direct evidence, contextual evidence, or only a lead.

## Maintainer Review

Read `references/reviewer-guide.md` when reviewing this contribution. It explains the offline smoke test, the AskNews path, and why the PR intentionally leaves workshop exercise files untouched.

## Skill Wrapper

If this skill folder is installed outside the repo, use the wrapper script and pass the repo path:

```bash
python scripts/research_brief.py --repo /path/to/Multimodal-search-workshop "find signs of an AI investment bubble"
```
