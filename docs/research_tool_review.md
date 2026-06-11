# Research Tool Review Guide

This PR adds a small user-facing research layer on top of the earnings-call
workshop corpus. It is intended to show how the exercise tools can become a
practical data-science and journalism workflow: a reviewer gives a free-text
research question, receives a finished analysis, and gets an evidence package
with transcript citations, audio clip paths, structured data, and optional live
AskNews context.

## Why This Is Valuable

- It turns the MCP search primitives into a complete reviewer-facing workflow.
- It works without credentials in transcript-only mode, so maintainers can test
  it quickly.
- When `ASKNEWS_API_KEY` is available, it adds a high-fidelity DeepNews pass that
  pressure-tests transcript findings against current external context.
- Every run writes a reproducible evidence package instead of only printing a
  transient answer.
- The same implementation is available through CLI, MCP, and a Codex Skill.

## Two-Minute Smoke Test

Run this from the repo root. It avoids Qdrant, Gemini, and AskNews so it should
work in a fresh local review environment after dependencies are installed:

```bash
.venv/bin/python -m research.cli \
  --no-qdrant \
  --no-asknews \
  --max-evidence 3 \
  "whats the common theme of the calls available? and whats the industry prognosis?"
```

Expected result:

- The terminal prints a finished `Industry read from available earnings calls`
  analysis, not just a ranked evidence table.
- Progress messages appear on stderr with a `[research]` prefix.
- The final line points to `research_outputs/<timestamp>_<task>/`.
- That package contains `README.md`, `analysis.md`, `evidence.md`,
  `evidence.json`, `asknews_context.md`, `asknews_context.json`, and
  `manifest.json`.

## AskNews Review Path

If `ASKNEWS_API_KEY` is set, run the same workflow without `--no-asknews`:

```bash
.venv/bin/python -m research.cli \
  --no-qdrant \
  --max-evidence 3 \
  --asknews-timeout 120 \
  "research cyclic investment in and by AI companies and find evidence for it in earnings calls"
```

Expected result:

- The transcript analysis still appears first.
- A `Live External Context (AskNews)` section is appended.
- The full external memo and cited source list are saved in
  `asknews_context.md` and `asknews_context.json`.
- If the network or key fails, the tool still returns the transcript analysis and
  records the AskNews error in the evidence package.

## MCP Review Path

The MCP server exposes the same workflow as:

```python
research_earnings(
    task_description="compare AI investment language across NVDA, AMZN, AAPL, and TSLA",
    max_evidence=8,
    use_semantic_search=True,
    use_asknews=True,
)
```

This is useful for Claude/Codex-style agents because the tool returns both
`analysis_markdown` and structured evidence objects.

## What To Inspect

- `analysis.md`: Is the output a readable answer to the user's question?
- `evidence.md`: Are transcript quotes tied to speakers, tickers, point IDs, and
  audio clips?
- `asknews_context.md`: Does live external context validate, contradict, or
  sharpen the transcript-only analysis?
- `README.md`: Can a reviewer understand the package without reading code?
- `manifest.json`: Are all side-effect artifacts discoverable?

## Review Caveats

- The workshop corpus is intentionally small, so this is not a complete market
  study.
- Transcript chunks can contain transcription errors; verify audio clips before
  publication.
- AskNews provides external context and source leads, but transcript evidence
  remains the core support for claims about what was said on calls.
