"""Build evidence-backed research briefs from earnings call transcripts."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = REPO_ROOT / "data" / "transcripts"
AUDIO_CLIPS_DIR = REPO_ROOT / "data" / "audio_clips"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research_outputs"
COLLECTION_TICKERS = {"AAPL", "AMZN", "NVDA", "TSLA"}
ASKNEWS_MODEL = "claude-sonnet-4-6"
ASKNEWS_SOURCES = ["asknews", "google", "x", "wiki"]

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

COMPANY_ALIASES = {
    "AAPL": {"aapl", "apple"},
    "AMZN": {"amzn", "amazon", "aws"},
    "NVDA": {"nvda", "nvidia"},
    "TSLA": {"tsla", "tesla"},
}

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "analysis",
    "an",
    "and",
    "any",
    "are",
    "around",
    "as",
    "at",
    "be",
    "because",
    "been",
    "being",
    "between",
    "both",
    "brief",
    "but",
    "by",
    "call",
    "calls",
    "can",
    "companies",
    "company",
    "could",
    "did",
    "does",
    "during",
    "earnings",
    "evidence",
    "find",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "like",
    "look",
    "make",
    "more",
    "not",
    "of",
    "on",
    "or",
    "our",
    "out",
    "over",
    "research",
    "said",
    "say",
    "show",
    "that",
    "the",
    "their",
    "there",
    "this",
    "through",
    "topic",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "you",
    "your",
}

EXPANSION_RULES = {
    "ai": (
        "artificial intelligence",
        "accelerated computing",
        "data center",
        "GPU",
        "compute",
        "training",
        "inference",
    ),
    "investment": (
        "capital expenditure",
        "capex",
        "infrastructure spending",
        "capacity",
        "supply",
        "demand",
    ),
    "capex": (
        "capital expenditure",
        "infrastructure spending",
        "capacity",
        "supply",
        "demand",
    ),
    "bubble": (
        "cyclical investment",
        "overinvestment",
        "demand digestion",
        "return on investment",
        "capacity cycle",
    ),
    "cyclicality": (
        "cyclical investment",
        "demand cycle",
        "supply cycle",
        "capacity cycle",
        "inventory",
    ),
    "cycle": (
        "cyclical investment",
        "demand cycle",
        "supply cycle",
        "inventory",
        "utilization",
    ),
    "tariff": (
        "tariffs",
        "supply chain",
        "gross margin",
        "pricing",
        "cost pressure",
    ),
}

THEME_DEFINITIONS = [
    {
        "id": "ai_infrastructure",
        "title": "AI infrastructure is the central growth narrative",
        "terms": (
            "AI",
            "artificial intelligence",
            "gen ai",
            "generative ai",
            "accelerated computing",
            "data center",
            "GPU",
            "Blackwell",
            "training",
            "inference",
            "foundation model",
            "AWS",
            "Bedrock",
            "Nova",
        ),
        "read": (
            "The calls repeatedly frame AI as an infrastructure and product platform shift, "
            "not merely a feature cycle."
        ),
    },
    {
        "id": "capacity_capex",
        "title": "Capacity, capex, and supply are the limiting factors",
        "terms": (
            "capex",
            "capital expenditure",
            "capacity",
            "infrastructure",
            "supply",
            "demand",
            "shortage",
            "constrained",
            "data center",
            "supply chain",
        ),
        "read": (
            "Management teams describe demand as strong, but the practical question is "
            "how quickly capacity, components, and infrastructure can catch up."
        ),
    },
    {
        "id": "macro_tariffs",
        "title": "Macro, tariffs, and geopolitics are near-term headwinds",
        "terms": (
            "tariff",
            "tariffs",
            "China",
            "macro",
            "headwind",
            "uncertainty",
            "consumer spending",
            "supply chain",
            "foreign exchange",
            "regulatory",
        ),
        "read": (
            "The near-term risk language is mostly about policy, trade, consumer demand, "
            "and supply-chain exposure rather than lack of long-run technology demand."
        ),
    },
    {
        "id": "margin_efficiency",
        "title": "Profitability depends on efficiency and price/performance",
        "terms": (
            "margin",
            "gross margin",
            "operating income",
            "profitability",
            "efficiency",
            "cost",
            "price performance",
            "free cash flow",
            "productivity",
        ),
        "read": (
            "The prognosis is not simply more spending; the calls tie upside to better "
            "unit economics, automation, and disciplined execution."
        ),
    },
    {
        "id": "autonomy_energy",
        "title": "Autonomy and electrification remain high-upside but uneven",
        "terms": (
            "autonomy",
            "autonomous",
            "robotaxi",
            "FSD",
            "vehicle",
            "vehicles",
            "EV",
            "energy storage",
            "Megapack",
            "battery",
        ),
        "read": (
            "Tesla-specific evidence points to long-run optionality in autonomy and energy, "
            "while the near-term vehicle market remains exposed to transition and demand risk."
        ),
    },
    {
        "id": "consumer_platforms",
        "title": "Consumer platforms are trying to turn AI into services and retention",
        "terms": (
            "services",
            "Alexa",
            "Prime",
            "advertising",
            "iPhone",
            "App Store",
            "customer",
            "consumer",
            "subscription",
            "retail",
        ),
        "read": (
            "Apple and Amazon language suggests the platform companies are translating AI "
            "and ecosystem scale into services, engagement, and customer retention."
        ),
    },
]

THEME_ORDER = {
    "ai_infrastructure": 0,
    "capacity_capex": 1,
    "macro_tariffs": 2,
    "margin_efficiency": 3,
    "consumer_platforms": 4,
    "autonomy_energy": 5,
}


@dataclass(frozen=True)
class ResearchPlan:
    task: str
    tickers: list[str]
    date_range: str | None
    keywords: list[str]
    queries: list[str]


def build_research_brief(
    task: str,
    *,
    max_evidence: int = 12,
    use_qdrant: bool = True,
    max_queries: int = 5,
) -> dict[str, Any]:
    """Return a JSON-serializable research brief for a free-text task."""

    plan = build_research_plan(task, max_queries=max_queries)
    local_chunks = load_local_chunks()
    errors: list[str] = []

    semantic_rows: list[dict[str, Any]] = []
    if use_qdrant:
        semantic_rows, errors = _semantic_candidates(plan)

    lexical_rows = _lexical_candidates(plan, local_chunks, limit=max(25, max_evidence * 4))
    evidence = _merge_and_rank_evidence(
        semantic_rows=semantic_rows,
        lexical_rows=lexical_rows,
        local_chunks=local_chunks,
        plan=plan,
        max_evidence=max_evidence,
    )

    brief = {
        "task": task,
        "plan": {
            "tickers": plan.tickers or sorted(COLLECTION_TICKERS),
            "date_range": plan.date_range,
            "keywords": plan.keywords,
            "queries": plan.queries,
            "semantic_search": use_qdrant,
        },
        "assessment": _assessment(evidence, plan),
        "findings": _findings(evidence, plan),
        "evidence": evidence,
        "limitations": _limitations(errors, use_qdrant),
        "next_steps": _next_steps(plan),
        "errors": errors,
    }
    brief["markdown"] = render_markdown(brief)
    return brief


def run_research_analysis(
    task: str,
    *,
    max_evidence: int = 12,
    use_qdrant: bool = True,
    max_queries: int = 5,
    output_dir: Path | None = None,
    write_package: bool = True,
    use_asknews: bool = True,
    asknews_timeout: float = 120.0,
) -> dict[str, Any]:
    """Build a finished analysis and optionally persist its evidence package."""

    corpus_mode = is_corpus_question(task)
    chunks = load_local_chunks()
    brief = build_research_brief(
        task,
        max_evidence=max_evidence,
        use_qdrant=(use_qdrant and not corpus_mode),
        max_queries=max_queries,
    )

    if corpus_mode:
        theme_package = build_corpus_theme_package(chunks)
        analysis_markdown = render_corpus_analysis(task, theme_package)
        mode = "corpus_analysis"
    else:
        theme_package = None
        analysis_markdown = render_targeted_analysis(brief)
        mode = "targeted_evidence_analysis"

    asknews_context = build_asknews_research_context(
        task=task,
        transcript_analysis=analysis_markdown,
        brief=brief,
        theme_package=theme_package,
        enabled=use_asknews,
        timeout=asknews_timeout,
    )
    analysis_markdown = integrate_asknews_context(
        analysis_markdown,
        asknews_context,
    )

    result = {
        "task": task,
        "mode": mode,
        "analysis_markdown": analysis_markdown,
        "evidence_package": None,
        "brief": brief,
        "theme_package": theme_package,
        "asknews_context": asknews_context,
    }

    if write_package:
        package_path = write_evidence_package(
            result,
            output_dir=output_dir or DEFAULT_OUTPUT_DIR,
        )
        result["evidence_package"] = str(package_path)
        result["analysis_markdown"] = (
            analysis_markdown.rstrip()
            + "\n\n---\n\n"
            + f"Evidence package: `{package_path}`\n"
        )

    return result


def build_asknews_research_context(
    *,
    task: str,
    transcript_analysis: str,
    brief: dict[str, Any],
    theme_package: dict[str, Any] | None,
    enabled: bool,
    timeout: float,
) -> dict[str, Any]:
    """Ask DeepNews to validate and enrich the transcript-derived analysis."""

    if not enabled:
        return {
            "status": "disabled",
            "analysis": "",
            "articles": [],
            "note": "AskNews augmentation was disabled for this run.",
        }

    asknews_key = os.getenv("ASKNEWS_API_KEY", "")
    if not asknews_key:
        return {
            "status": "skipped",
            "analysis": "",
            "articles": [],
            "note": "ASKNEWS_API_KEY is not set; live AskNews augmentation was skipped.",
        }

    try:
        from asknews_sdk import AskNewsSDK  # type: ignore
        from asknews_sdk.dto.deepnews import (  # type: ignore
            AnthropicTextDelta,
            ContentBlockDeltaEvent,
            CreateDeepNewsResponseStreamChunkV2,
            CreateDeepNewsResponseStreamSource,
            CreateDeepNewsResponseStreamSourcesNewsSource,
            CreateDeepNewsResponseStreamSourcesWebSource,
        )

        ask = AskNewsSDK(api_key=asknews_key, timeout=timeout)
        query = _asknews_research_prompt(
            task=task,
            transcript_analysis=transcript_analysis,
            brief=brief,
            theme_package=theme_package,
        )

        response = ask.chat.get_deep_news(
            messages=[{"role": "user", "content": query}],
            model=ASKNEWS_MODEL,
            stream=True,
            inline_citations="markdown_link",
            append_references=True,
            journalist_mode=True,
            sources=ASKNEWS_SOURCES,
            search_depth=2,
            max_depth=4,
            return_sources=True,
            include_entities=True,
            engine="v2.0",
            only_cited_sources=True,
            max_parallel_tool_calls=2,
            max_tokens=3500,
            temperature=0.2,
        )

        analysis_parts: list[str] = []
        articles: list[dict[str, Any]] = []
        seen_article_ids: set[str] = set()

        for message in response:
            if isinstance(message, CreateDeepNewsResponseStreamChunkV2):
                event = message.choices[0].delta
                if isinstance(event, ContentBlockDeltaEvent) and isinstance(
                    event.delta,
                    AnthropicTextDelta,
                ):
                    analysis_parts.append(event.delta.text)
                continue

            if not isinstance(message, CreateDeepNewsResponseStreamSource):
                continue

            if isinstance(message.source, CreateDeepNewsResponseStreamSourcesNewsSource):
                item = message.source.data
                article_id = str(item.article_id)
                if article_id in seen_article_ids:
                    continue
                articles.append(
                    {
                        "title": item.eng_title or item.title,
                        "summary": item.summary,
                        "source": item.source_id,
                        "url": str(item.article_url),
                        "published_at": str(item.pub_date),
                        "sentiment": item.sentiment,
                        "language": item.language,
                        "content_type": item.content_type,
                    }
                )
                seen_article_ids.add(article_id)

            elif isinstance(message.source, CreateDeepNewsResponseStreamSourcesWebSource):
                item = message.source.data
                article_id = str(item.url)
                if article_id in seen_article_ids:
                    continue
                articles.append(
                    {
                        "title": item.title,
                        "summary": (
                            " ".join(item.key_points)
                            if item.key_points
                            else item.raw_text
                        ),
                        "source": item.source,
                        "url": str(item.url),
                        "published_at": item.published,
                        "sentiment": None,
                        "language": "",
                        "content_type": "web",
                    }
                )
                seen_article_ids.add(article_id)

        return {
            "status": "ok",
            "model": ASKNEWS_MODEL,
            "sources": ASKNEWS_SOURCES,
            "analysis": _clean_asknews_analysis("".join(analysis_parts)),
            "articles": articles,
            "query": query,
        }
    except Exception as exc:
        return {
            "status": "error",
            "analysis": "",
            "articles": [],
            "error": str(exc),
            "note": "AskNews augmentation failed; transcript-only analysis is still available.",
        }


def integrate_asknews_context(
    analysis_markdown: str,
    asknews_context: dict[str, Any],
) -> str:
    """Append a concise live-news section to the finished analysis."""

    status = asknews_context.get("status")
    if status == "ok" and asknews_context.get("analysis"):
        articles = asknews_context.get("articles", [])
        lines = [
            analysis_markdown.rstrip(),
            "",
            "## Live External Context (AskNews)",
            "",
            _truncate_markdown(str(asknews_context["analysis"]), limit=1800),
            "",
            f"AskNews returned {len(articles)} cited source records from {', '.join(ASKNEWS_SOURCES)}.",
        ]
        if articles:
            lines.extend(["", "Top cited sources:"])
            for article in articles[:5]:
                title = article.get("title") or "Untitled"
                source = article.get("source") or "unknown source"
                published = str(article.get("published_at") or "")[:10]
                url = article.get("url") or ""
                suffix = f" ({published})" if published else ""
                link = f" - {url}" if url else ""
                lines.append(f"- {title} - {source}{suffix}{link}")
        return "\n".join(lines) + "\n"

    note = asknews_context.get("note") or asknews_context.get("error")
    if note:
        return (
            analysis_markdown.rstrip()
            + "\n\n## Live External Context (AskNews)\n\n"
            + f"{note}\n"
        )
    return analysis_markdown


def is_corpus_question(task: str) -> bool:
    """Detect prompts that ask for synthesis across the available calls."""

    lower = task.lower()
    corpus_markers = (
        "common theme",
        "common themes",
        "theme of the calls",
        "themes of the calls",
        "calls available",
        "available calls",
        "industry prognosis",
        "industry outlook",
        "overall prognosis",
        "overall outlook",
        "across the calls",
        "across calls",
        "summarize the calls",
        "what do the calls say",
    )
    return any(marker in lower for marker in corpus_markers)


def build_corpus_theme_package(chunks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Score the whole corpus against stable business themes."""

    calls = _call_summaries(chunks)
    themes = []
    for theme in THEME_DEFINITIONS:
        evidence = _theme_evidence(chunks, theme, limit=5)
        if not evidence:
            continue
        tickers = Counter(row["ticker"] for row in evidence if row.get("ticker"))
        themes.append(
            {
                "id": theme["id"],
                "title": theme["title"],
                "read": theme["read"],
                "evidence_count": len(evidence),
                "tickers": dict(tickers.most_common()),
                "evidence": evidence,
            }
        )

    themes.sort(
        key=lambda t: (
            THEME_ORDER.get(t["id"], 99),
            -len(t["tickers"]),
            -sum(row["score"] for row in t["evidence"]),
        )
    )
    return {
        "calls": calls,
        "themes": themes,
        "corpus_size": {
            "calls": len(calls),
            "chunks": len(chunks),
            "words": sum(call["word_count"] for call in calls),
        },
    }


def render_corpus_analysis(task: str, package: dict[str, Any]) -> str:
    """Render a finished corpus-level analysis for a broad research prompt."""

    calls = package["calls"]
    themes = package["themes"]
    leading = themes[:4]
    ai_theme = _theme_by_id(themes, "ai_infrastructure")
    capex_theme = _theme_by_id(themes, "capacity_capex")
    macro_theme = _theme_by_id(themes, "macro_tariffs")
    margin_theme = _theme_by_id(themes, "margin_efficiency")

    lines = [
        "# Industry read from available earnings calls",
        "",
        "## Bottom Line",
        "",
        (
            "Across the available calls, the common theme is that AI and automation are "
            "becoming the organizing growth narrative, but the investable prognosis is "
            "uneven: infrastructure demand looks strong, while capacity, trade policy, "
            "consumer demand, and execution risk decide who captures the upside."
        ),
        "",
        "The corpus is small, so this should be read as a directional evidence brief, "
        "not a complete industry survey. The calls in scope are "
        + ", ".join(
            f"{call['ticker']} {call['quarter']} {call['year']}"
            for call in calls
        )
        + ".",
        "",
        "## Common Themes",
        "",
    ]

    for idx, theme in enumerate(leading, start=1):
        tickers = ", ".join(theme["tickers"].keys())
        lines.append(f"{idx}. **{theme['title']}.** {theme['read']} Evidence appears in {tickers}.")
        for row in theme["evidence"][:2]:
            lines.append(
                f"   - {row['ticker']} ({row['speaker']}): \"{row['quote']}\""
            )
        lines.append("")

    lines.extend(
        [
            "## Industry Prognosis",
            "",
            (
                "- **AI infrastructure/cloud/semis: positive but capacity-constrained.** "
                + _prognosis_sentence(ai_theme, capex_theme)
            ),
            (
                "- **Platform companies: AI is becoming a retention and services layer.** "
                "Amazon and Apple-style platform economics matter because AI demand is being "
                "packaged into cloud services, assistants, advertising, subscriptions, and devices."
            ),
            (
                "- **EV/autonomy: higher variance.** Tesla evidence points to autonomy, energy "
                "storage, and manufacturing transition as upside cases, but near-term demand and "
                "policy exposure make the forecast less clean than for AI infrastructure."
            ),
            (
                "- **Risk regime: policy and ROI scrutiny.** "
                + _risk_sentence(macro_theme, margin_theme)
            ),
            "",
            "## What To Watch Next",
            "",
            "- Whether AI capex converts into revenue and cash flow, not only capacity announcements.",
            "- Whether supply-chain constraints ease without creating overcapacity.",
            "- Whether tariffs or export controls change gross-margin guidance.",
            "- Whether consumer-facing AI products show pricing power rather than just engagement.",
            "",
            "## Caveats",
            "",
            "- The evidence package contains chunk-level transcript excerpts and audio clip paths; verify quotes before publishing.",
            "- This analysis uses the workshop corpus only and should be expanded with filings, segment data, and later calls.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_targeted_analysis(brief: dict[str, Any]) -> str:
    """Render a finished analysis for a focused hypothesis/evidence prompt."""

    evidence = brief["evidence"]
    lines = [
        "# Earnings-call analysis",
        "",
        f"Research question: {brief['task']}",
        "",
        "## Answer",
        "",
    ]
    if not evidence:
        lines.append(
            "The available calls do not provide strong evidence for the prompt. Treat this "
            "as an absence of retrieved support, not proof that the claim is false."
        )
    else:
        direct = sum(1 for row in evidence if row["strength"] == "direct")
        contextual = sum(1 for row in evidence if row["strength"] == "contextual")
        tickers = Counter(row["ticker"] for row in evidence if row.get("ticker"))
        lines.append(
            f"The retrieved evidence is {'moderate' if direct else 'weak-to-moderate'}: "
            f"{direct} direct chunks and {contextual} contextual chunks, concentrated in "
            f"{', '.join(tickers.keys())}."
        )
    lines.extend(["", "## Main Findings", ""])
    lines.extend(f"- {finding}" for finding in brief["findings"])
    lines.extend(["", "## Best Evidence", ""])
    for row in evidence[:5]:
        lines.append(
            f"- **{row['ticker']} {row.get('quarter', '')} {row.get('year', '')} "
            f"({row['speaker']})**: \"{row['quote']}\""
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {item}" for item in brief["limitations"])
    return "\n".join(lines) + "\n"


def write_evidence_package(result: dict[str, Any], *, output_dir: Path) -> Path:
    """Persist analysis and evidence artifacts and return the package path."""

    package_dir = output_dir / _package_slug(result["task"])
    package_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = package_dir / "analysis.md"
    evidence_md_path = package_dir / "evidence.md"
    evidence_json_path = package_dir / "evidence.json"
    asknews_md_path = package_dir / "asknews_context.md"
    asknews_json_path = package_dir / "asknews_context.json"
    manifest_path = package_dir / "manifest.json"

    analysis_path.write_text(result["analysis_markdown"])
    if result["theme_package"]:
        evidence_md_path.write_text(render_theme_evidence_markdown(result["theme_package"]))
    else:
        evidence_md_path.write_text(result["brief"]["markdown"])
    evidence_json_path.write_text(
        json.dumps(
            {
                "task": result["task"],
                "mode": result["mode"],
                "brief": result["brief"],
                "theme_package": result["theme_package"],
                "asknews_context": result["asknews_context"],
            },
            indent=2,
        )
    )
    asknews_md_path.write_text(render_asknews_markdown(result["asknews_context"]))
    asknews_json_path.write_text(json.dumps(result["asknews_context"], indent=2))
    manifest_path.write_text(
        json.dumps(
            {
                "task": result["task"],
                "mode": result["mode"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": {
                    "analysis": str(analysis_path),
                    "evidence_markdown": str(evidence_md_path),
                    "evidence_json": str(evidence_json_path),
                    "asknews_markdown": str(asknews_md_path),
                    "asknews_json": str(asknews_json_path),
                },
            },
            indent=2,
        )
    )
    return package_dir


def render_asknews_markdown(asknews_context: dict[str, Any]) -> str:
    lines = [
        "# AskNews Live Context",
        "",
        f"Status: `{asknews_context.get('status', 'unknown')}`",
        "",
    ]
    if asknews_context.get("analysis"):
        lines.extend(["## Analysis", "", str(asknews_context["analysis"]).strip(), ""])
    if asknews_context.get("note"):
        lines.extend(["## Note", "", str(asknews_context["note"]), ""])
    if asknews_context.get("error"):
        lines.extend(["## Error", "", str(asknews_context["error"]), ""])
    articles = asknews_context.get("articles") or []
    if articles:
        lines.extend(["## Sources", ""])
        for article in articles:
            title = article.get("title") or "Untitled"
            source = article.get("source") or "unknown source"
            published = str(article.get("published_at") or "")[:10]
            url = article.get("url") or ""
            lines.append(f"- {title} - {source} - {published}")
            if url:
                lines.append(f"  - {url}")
            if article.get("summary"):
                lines.append(f"  - {str(article['summary'])[:500]}")
    return "\n".join(lines).rstrip() + "\n"


def _asknews_research_prompt(
    *,
    task: str,
    transcript_analysis: str,
    brief: dict[str, Any],
    theme_package: dict[str, Any] | None,
) -> str:
    evidence_lines: list[str] = []
    if theme_package:
        for theme in theme_package.get("themes", [])[:5]:
            evidence_lines.append(f"Theme: {theme['title']}")
            for row in theme.get("evidence", [])[:3]:
                evidence_lines.append(
                    f"- {row.get('ticker')} {row.get('quarter')} {row.get('year')} "
                    f"{row.get('speaker')}: {row.get('quote')}"
                )
    else:
        for row in brief.get("evidence", [])[:10]:
            evidence_lines.append(
                f"- {row.get('ticker')} {row.get('quarter')} {row.get('year')} "
                f"{row.get('speaker')}: {row.get('quote')}"
            )

    return (
        "You are a high-fidelity financial research assistant for a journalist/data scientist.\n"
        "Use live AskNews, web, X/Twitter, and Wikipedia sources to pressure-test and enrich "
        "the transcript-derived analysis below. Focus on current and recent external context, "
        "company-specific developments, macro policy, AI capex, demand/supply constraints, "
        "tariffs/export controls, and industry outlook.\n\n"
        "Return only the finished cited research memo. Do not narrate your search process, "
        "do not include XML/HTML wrapper tags, and do not include a preamble. Start directly "
        "with `## External Validation` and include these sections:\n"
        "- External Validation: which transcript themes are supported by live outside evidence.\n"
        "- Contradictions or Missing Context: what the transcript-only analysis may miss.\n"
        "- Updated Industry Prognosis: what changes after adding live context.\n"
        "- Source Notes: cite the most important sources inline.\n\n"
        f"User task:\n{task}\n\n"
        f"Transcript-derived analysis:\n{_truncate_markdown(transcript_analysis, limit=3000)}\n\n"
        "Transcript evidence snippets:\n"
        + "\n".join(evidence_lines[:24])
    )


def build_research_plan(task: str, *, max_queries: int = 5) -> ResearchPlan:
    """Infer a small search plan from a research prompt."""

    cleaned = " ".join(task.split())
    keywords = extract_keywords(cleaned)
    tickers = infer_tickers(cleaned)
    date_range = infer_date_range(cleaned)

    queries: list[str] = [cleaned]
    quoted = [p.strip() for p in re.findall(r'"([^"]+)"|\'([^\']+)\'', cleaned) for p in p if p]
    queries.extend(q for q in quoted if len(q) > 3)

    if keywords:
        queries.append(_keyword_query(keywords))

    lower_terms = set(keywords) | set(_tokens(cleaned))
    expansions: list[str] = []
    for trigger, phrases in EXPANSION_RULES.items():
        if trigger in lower_terms or (trigger == "ai" and "artificial" in lower_terms):
            expansions.extend(phrases)
    if expansions:
        queries.append(" ".join(dict.fromkeys(expansions)))

    if {"bubble", "overinvestment", "cyclical", "cycle"} & lower_terms:
        queries.append(
            "capex capacity demand supply ROI utilization digestion inventory risk"
        )

    return ResearchPlan(
        task=cleaned,
        tickers=tickers,
        date_range=date_range,
        keywords=keywords,
        queries=_dedupe(queries)[:max_queries],
    )


def extract_keywords(text: str, *, limit: int = 14) -> list[str]:
    """Extract compact topic terms for search planning and evidence scoring."""

    counts: Counter[str] = Counter()
    for tok in _tokens(text):
        if tok in STOPWORDS or tok in COMPANY_ALIASES:
            continue
        if tok.isdigit():
            continue
        counts[tok] += 1

    for phrase in re.findall(r"\b[A-Za-z][A-Za-z0-9-]+(?:\s+[A-Za-z][A-Za-z0-9-]+){1,3}\b", text):
        words = [w for w in _tokens(phrase) if w not in STOPWORDS]
        if 2 <= len(words) <= 4:
            counts[" ".join(words)] += 2

    return [term for term, _ in counts.most_common(limit)]


def infer_tickers(text: str) -> list[str]:
    """Find known tickers/company names mentioned in the prompt."""

    lower = text.lower()
    tickers = set()
    for ticker in COLLECTION_TICKERS:
        if re.search(rf"\b{ticker}\b", text, flags=re.IGNORECASE):
            tickers.add(ticker)
    for ticker, aliases in COMPANY_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", lower) for alias in aliases):
            tickers.add(ticker)
    return sorted(tickers)


def infer_date_range(text: str) -> str | None:
    """Infer a simple ISO date range from years or explicit dates."""

    dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if len(dates) >= 2:
        return f"{min(dates)}:{max(dates)}"

    years = sorted({int(y) for y in re.findall(r"\b(20\d{2})\b", text)})
    if len(years) >= 2:
        return f"{years[0]}-01-01:{years[-1]}-12-31"
    if len(years) == 1:
        return f"{years[0]}-01-01:{years[0]}-12-31"
    return None


def load_local_chunks() -> dict[str, dict[str, Any]]:
    """Load transcript chunks and enrich them with stable point IDs when known."""

    point_map_path = TRANSCRIPTS_DIR / "point_map.json"
    point_map = json.loads(point_map_path.read_text()) if point_map_path.exists() else {}
    by_audio_offset: dict[tuple[str, float, float], str] = {}
    for point_id, meta in point_map.items():
        key = (
            str(meta.get("audio_file", "")),
            round(float(meta.get("start_time", 0.0)), 2),
            round(float(meta.get("end_time", 0.0)), 2),
        )
        by_audio_offset[key] = point_id

    chunks: dict[str, dict[str, Any]] = {}
    for path in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        if path.name == "point_map.json":
            continue
        data = json.loads(path.read_text())
        audio_file = data.get("audio_file", "")
        for chunk in data.get("chunks", []):
            start_time = float(chunk.get("start", 0.0))
            end_time = float(chunk.get("end", 0.0))
            key = (audio_file, round(start_time, 2), round(end_time, 2))
            point_id = by_audio_offset.get(key)
            fallback_id = f"{audio_file}:{chunk.get('chunk_index', len(chunks))}"
            row = {
                "point_id": point_id or fallback_id,
                "has_point_id": bool(point_id),
                "ticker": data.get("ticker"),
                "company": data.get("company"),
                "quarter": data.get("quarter"),
                "year": data.get("year"),
                "date": data.get("date"),
                "chunk_index": chunk.get("chunk_index"),
                "chunk_text": chunk.get("text", ""),
                "speaker": chunk.get("speaker", ""),
                "start_time": start_time,
                "end_time": end_time,
                "audio_file": audio_file,
            }
            chunks[row["point_id"]] = row
    return chunks


def render_markdown(brief: dict[str, Any]) -> str:
    """Render a research brief as Markdown."""

    plan = brief["plan"]
    lines = [
        "# Earnings-call research brief",
        "",
        f"Task: {brief['task']}",
        "",
        "## Assessment",
        "",
        brief["assessment"],
        "",
        "## Method",
        "",
        f"- Universe: {', '.join(plan['tickers'])}",
        f"- Date range: {plan['date_range'] or 'all available calls'}",
        f"- Search mode: {'semantic + lexical' if plan['semantic_search'] else 'local lexical only'}",
        "- Query plan:",
    ]
    lines.extend(f"  - {q}" for q in plan["queries"])
    lines.extend(["", "## Findings", ""])
    lines.extend(f"- {finding}" for finding in brief["findings"])
    lines.extend(["", "## Evidence", ""])

    for idx, row in enumerate(brief["evidence"], start=1):
        when = " ".join(str(v) for v in [row.get("quarter"), row.get("year")] if v)
        date = f", {row['date']}" if row.get("date") else ""
        speaker = row.get("speaker") or "unknown speaker"
        lines.append(
            f"{idx}. {row['strength'].upper()} - {row['ticker']} {when}{date} - {speaker}"
        )
        lines.append(f"   - Score: {row['score']:.3f}; matched: {', '.join(row['matched_terms']) or 'semantic similarity'}")
        lines.append(f"   - Quote: \"{row['quote']}\"")
        lines.append(f"   - Point ID: `{row['point_id']}`")
        if row.get("audio_clip"):
            lines.append(f"   - Audio clip: `{row['audio_clip']}`")
        lines.append("")

    lines.extend(["## Limitations", ""])
    lines.extend(f"- {item}" for item in brief["limitations"])
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {item}" for item in brief["next_steps"])
    lines.append("")
    return "\n".join(lines)


def _semantic_candidates(plan: ResearchPlan) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    targets = plan.tickers or [None]

    try:
        from mcp_server.server import search_earnings
    except Exception as exc:
        return [], [f"Could not load MCP search tool: {exc}"]

    for query in plan.queries:
        for ticker in targets:
            result = search_earnings(
                query=query,
                ticker=ticker,
                date_range=plan.date_range,
                boost_recency=_wants_recency(plan.task),
            )
            if result and "error" in result[0]:
                label = f"{ticker or 'all'} / {query}"
                errors.append(f"Semantic search failed for {label}: {result[0]['error']}")
                continue
            for rank, row in enumerate(result, start=1):
                enriched = dict(row)
                enriched["source_query"] = query
                enriched["semantic_rank"] = rank
                enriched["semantic_score"] = float(row.get("score") or 0.0)
                rows.append(enriched)
    return rows, errors


def _lexical_candidates(
    plan: ResearchPlan,
    chunks: dict[str, dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    terms = _score_terms(plan)
    rows: list[dict[str, Any]] = []
    allowed = set(plan.tickers) if plan.tickers else None

    for row in chunks.values():
        if allowed and row.get("ticker") not in allowed:
            continue
        if plan.date_range and not _date_in_range(row.get("date"), plan.date_range):
            continue

        text = row.get("chunk_text", "")
        score, matched = _text_score(text, terms)
        if score <= 0:
            continue
        candidate = dict(row)
        candidate["lexical_score"] = score
        candidate["matched_terms"] = matched
        rows.append(candidate)

    rows.sort(key=lambda r: (r["lexical_score"], _date_sort_value(r.get("date"))), reverse=True)
    return rows[:limit]


def _merge_and_rank_evidence(
    *,
    semantic_rows: list[dict[str, Any]],
    lexical_rows: list[dict[str, Any]],
    local_chunks: dict[str, dict[str, Any]],
    plan: ResearchPlan,
    max_evidence: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for row in lexical_rows:
        merged[row["point_id"]] = dict(row)

    for row in semantic_rows:
        point_id = str(row.get("point_id"))
        base = dict(local_chunks.get(point_id, {}))
        base.update({k: v for k, v in row.items() if v is not None})
        current = merged.get(point_id, {})
        current.update(base)
        current["semantic_score"] = max(
            float(current.get("semantic_score") or 0.0),
            float(row.get("semantic_score") or row.get("score") or 0.0),
        )
        current["source_query"] = row.get("source_query") or current.get("source_query")
        merged[point_id] = current

    evidence: list[dict[str, Any]] = []
    terms = _score_terms(plan)
    for point_id, row in merged.items():
        text = row.get("chunk_text", "")
        lexical_score, matched_terms = _text_score(text, terms)
        semantic_score = float(row.get("semantic_score") or row.get("score") or 0.0)
        combined = _combined_score(semantic_score, lexical_score, row.get("date"))
        quote = _quote_for_terms(text, matched_terms or plan.keywords)
        audio_clip = AUDIO_CLIPS_DIR / f"{point_id}.mp3"
        evidence.append(
            {
                "point_id": point_id,
                "ticker": row.get("ticker"),
                "company": row.get("company"),
                "quarter": row.get("quarter"),
                "year": row.get("year"),
                "date": row.get("date"),
                "speaker": row.get("speaker"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "score": combined,
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "matched_terms": matched_terms,
                "strength": _strength(semantic_score, lexical_score, matched_terms),
                "quote": quote,
                "source_query": row.get("source_query"),
                "audio_clip": str(audio_clip) if audio_clip.exists() else None,
            }
        )

    evidence.sort(
        key=lambda r: (
            {"direct": 3, "contextual": 2, "lead": 1}.get(r["strength"], 0),
            r["score"],
            _date_sort_value(r.get("date")),
        ),
        reverse=True,
    )
    return evidence[:max_evidence]


def _assessment(evidence: list[dict[str, Any]], plan: ResearchPlan) -> str:
    if not evidence:
        return (
            "No transcript evidence matched strongly enough. Treat this as a search miss, "
            "not as proof the topic is absent."
        )

    counts = Counter(row["strength"] for row in evidence)
    tickers = Counter(row.get("ticker") for row in evidence if row.get("ticker"))
    top_tickers = ", ".join(f"{ticker} ({count})" for ticker, count in tickers.most_common(4))
    return (
        f"Found {len(evidence)} candidate evidence chunks: "
        f"{counts.get('direct', 0)} direct, {counts.get('contextual', 0)} contextual, "
        f"{counts.get('lead', 0)} weaker leads. Most evidence is from {top_tickers or 'the available universe'}."
    )


def _findings(evidence: list[dict[str, Any]], plan: ResearchPlan) -> list[str]:
    if not evidence:
        return ["No findings yet; broaden queries, add company aliases, or inspect full transcripts."]

    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        by_ticker[str(row.get("ticker") or "UNKNOWN")].append(row)

    findings: list[str] = []
    for ticker, rows in sorted(by_ticker.items(), key=lambda item: len(item[1]), reverse=True)[:4]:
        terms = Counter(term for row in rows for term in row.get("matched_terms", []))
        term_text = ", ".join(term for term, _ in terms.most_common(5)) or "semantic matches"
        direct = sum(1 for row in rows if row["strength"] == "direct")
        findings.append(
            f"{ticker}: {len(rows)} evidence chunks ({direct} direct) cluster around {term_text}."
        )

    if any(row["strength"] == "lead" for row in evidence):
        findings.append(
            "Some high-ranked chunks are leads rather than proof; use the audio and full transcript to verify context."
        )
    return findings


def _limitations(errors: list[str], use_qdrant: bool) -> list[str]:
    limitations = [
        "This is retrieval evidence from a small workshop corpus, not a complete market study.",
        "Quotes are transcript chunks; verify the audio clip before publishing.",
        "The tool ranks evidence candidates and does not prove causality by itself.",
    ]
    if not use_qdrant:
        limitations.append("Semantic Qdrant search was disabled; results are keyword-only.")
    if errors:
        limitations.append("Some semantic searches failed; local transcript scoring filled the gap.")
    return limitations


def _next_steps(plan: ResearchPlan) -> list[str]:
    steps = [
        "Open the cited point IDs with get_audio_clip before using any quote.",
        "Search for counter-evidence using terms like risk, slowdown, digestion, ROI, and demand normalization.",
        "Compare the earnings-call claims with capex, revenue growth, and cash-flow data from filings.",
    ]
    if not plan.tickers:
        steps.append("Run a narrower pass by ticker once the strongest companies are identified.")
    return steps


def _call_summaries(chunks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in chunks.values():
        grouped[
            (
                str(row.get("ticker") or ""),
                str(row.get("quarter") or ""),
                int(row.get("year") or 0),
                str(row.get("date") or ""),
            )
        ].append(row)

    calls = []
    for (ticker, quarter, year, date), rows in grouped.items():
        calls.append(
            {
                "ticker": ticker,
                "company": rows[0].get("company"),
                "quarter": quarter,
                "year": year,
                "date": date,
                "chunks": len(rows),
                "word_count": sum(
                    len(str(row.get("chunk_text") or "").split()) for row in rows
                ),
            }
        )
    calls.sort(key=lambda row: (row["date"], row["ticker"]))
    return calls


def _theme_evidence(
    chunks: dict[str, dict[str, Any]],
    theme: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    terms = list(theme["terms"])
    rows = []
    for point_id, row in chunks.items():
        score, matched = _text_score(str(row.get("chunk_text") or ""), terms)
        if score <= 0:
            continue
        audio_clip = AUDIO_CLIPS_DIR / f"{point_id}.mp3"
        rows.append(
            {
                "point_id": point_id,
                "ticker": row.get("ticker"),
                "company": row.get("company"),
                "quarter": row.get("quarter"),
                "year": row.get("year"),
                "date": row.get("date"),
                "speaker": row.get("speaker"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "score": score,
                "matched_terms": matched,
                "quote": _quote_for_terms(str(row.get("chunk_text") or ""), matched),
                "audio_clip": str(audio_clip) if audio_clip.exists() else None,
            }
        )

    rows.sort(
        key=lambda row: (
            len(row["matched_terms"]),
            row["score"],
            _date_sort_value(row.get("date")),
        ),
        reverse=True,
    )
    management_rows = [row for row in rows if _is_management_speaker(row.get("speaker"))]
    return _diversify_by_ticker(management_rows or rows, limit=limit)


def render_theme_evidence_markdown(package: dict[str, Any]) -> str:
    lines = [
        "# Evidence Package",
        "",
        "## Corpus",
        "",
    ]
    for call in package["calls"]:
        lines.append(
            f"- {call['ticker']} {call['quarter']} {call['year']} "
            f"({call['date']}): {call['chunks']} chunks, {call['word_count']} words"
        )
    lines.extend(["", "## Theme Evidence", ""])
    for theme in package["themes"]:
        lines.append(f"### {theme['title']}")
        lines.append("")
        lines.append(theme["read"])
        lines.append("")
        for row in theme["evidence"]:
            when = f"{row.get('quarter', '')} {row.get('year', '')}".strip()
            lines.append(
                f"- **{row['ticker']} {when} - {row['speaker']}** "
                f"({', '.join(row['matched_terms'])})"
            )
            lines.append(f"  - Quote: \"{row['quote']}\"")
            lines.append(f"  - Point ID: `{row['point_id']}`")
            if row.get("audio_clip"):
                lines.append(f"  - Audio clip: `{row['audio_clip']}`")
        lines.append("")
    return "\n".join(lines)


def _is_management_speaker(speaker: Any) -> bool:
    speaker_text = str(speaker or "").lower()
    if not speaker_text:
        return False
    excluded = ("analyst", "operator", "external", "conference")
    return not any(term in speaker_text for term in excluded)


def _diversify_by_ticker(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    per_ticker: Counter[str] = Counter()
    for row in rows:
        ticker = str(row.get("ticker") or "")
        if per_ticker[ticker] >= 1:
            continue
        selected.append(row)
        per_ticker[ticker] += 1
        if len(selected) >= limit:
            return selected

    for row in rows:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _theme_by_id(themes: list[dict[str, Any]], theme_id: str) -> dict[str, Any] | None:
    for theme in themes:
        if theme["id"] == theme_id:
            return theme
    return None


def _prognosis_sentence(
    ai_theme: dict[str, Any] | None,
    capex_theme: dict[str, Any] | None,
) -> str:
    ai_tickers = ", ".join((ai_theme or {}).get("tickers", {}).keys())
    capex_tickers = ", ".join((capex_theme or {}).get("tickers", {}).keys())
    if ai_tickers and capex_tickers:
        return (
            f"The strongest evidence comes from {ai_tickers}, with capacity/capex "
            f"constraints visible in {capex_tickers}."
        )
    return (
        "The strongest evidence points to continued AI demand, but the corpus is too "
        "small to size the market cycle."
    )


def _risk_sentence(
    macro_theme: dict[str, Any] | None,
    margin_theme: dict[str, Any] | None,
) -> str:
    risk_tickers = ", ".join((macro_theme or {}).get("tickers", {}).keys())
    margin_tickers = ", ".join((margin_theme or {}).get("tickers", {}).keys())
    if risk_tickers and margin_tickers:
        return (
            f"Macro and tariff risks show up in {risk_tickers}; margin and efficiency "
            f"language shows up in {margin_tickers}."
        )
    return (
        "The core risk is that spending growth outruns monetization, especially if "
        "macro or policy conditions tighten."
    )


def _package_slug(task: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")
    return f"{stamp}_{slug[:64] or 'research'}"


def _score_terms(plan: ResearchPlan) -> list[str]:
    terms = list(plan.keywords)
    lower = set(_tokens(plan.task))
    for trigger, expansions in EXPANSION_RULES.items():
        if trigger in lower or trigger in terms:
            terms.extend(expansions)
    return _dedupe(terms)


def _keyword_query(keywords: list[str], *, limit: int = 9) -> str:
    phrases = [term for term in keywords if " " in term]
    phrase_words = {word for phrase in phrases for word in phrase.split()}
    singles = [
        term for term in keywords if " " not in term and term not in phrase_words
    ]
    return " ".join((phrases + singles)[:limit])


def _text_score(text: str, terms: list[str]) -> tuple[float, list[str]]:
    lower = text.lower()
    matched: list[str] = []
    score = 0.0
    for term in terms:
        needle = term.lower()
        if len(needle) < 3:
            continue
        if " " in needle:
            count = lower.count(needle)
            if count:
                matched.append(term)
                score += 2.5 * count
        else:
            count = len(re.findall(rf"\b{re.escape(needle)}s?\b", lower))
            if count:
                matched.append(term)
                score += float(count)
    return score, _dedupe(matched)


def _combined_score(semantic_score: float, lexical_score: float, date: str | None) -> float:
    recency = 0.0
    year = _date_sort_value(date)
    if year:
        recency = min(0.2, max(0.0, (year - 2023.0) * 0.05))
    return semantic_score + math.log1p(lexical_score) / 4.0 + recency


def _strength(semantic_score: float, lexical_score: float, matched_terms: list[str]) -> str:
    if lexical_score >= 4.0 and len(matched_terms) >= 2:
        return "direct"
    if semantic_score > 0.5 or lexical_score >= 2.0:
        return "contextual"
    return "lead"


def _quote_for_terms(text: str, terms: list[str], *, limit: int = 360) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean

    lower = clean.lower()
    positions = [lower.find(term.lower()) for term in terms if lower.find(term.lower()) >= 0]
    if not positions:
        return clean[: limit - 3].rstrip() + "..."

    center = min(positions)
    start = max(0, center - limit // 3)
    end = min(len(clean), start + limit)
    excerpt = clean[start:end].strip()
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(clean):
        excerpt += "..."
    return excerpt


def _truncate_markdown(text: str, *, limit: int) -> str:
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    excerpt = clean[: max(1, limit - 120)].rstrip()
    cut_points = [
        excerpt.rfind("\n\n"),
        excerpt.rfind(". "),
        excerpt.rfind("\n"),
    ]
    cut_at = max(cut_points)
    if cut_at > limit * 0.45:
        excerpt = excerpt[:cut_at].rstrip()
    return (
        excerpt
        + "\n\n[Full AskNews memo saved in `asknews_context.md` in the evidence package.]"
    )


def _clean_asknews_analysis(text: str) -> str:
    clean = text.strip()
    final_answer = re.search(r"<final_answer>\s*", clean, flags=re.IGNORECASE)
    if final_answer:
        clean = clean[final_answer.end() :]
    clean = re.sub(r"</?final_answer>\s*", "", clean, flags=re.IGNORECASE).strip()

    first_heading = re.search(r"(?m)^#{1,3}\s+", clean)
    if first_heading and 0 < first_heading.start() < 1200:
        clean = clean[first_heading.start() :].strip()

    return clean


def _tokens(text: str) -> list[str]:
    return [t.lower().strip("$.,:;()[]{}") for t in re.findall(r"[A-Za-z$][A-Za-z0-9$.-]*", text)]


def _dedupe(items: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = " ".join(str(item).lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(item))
    return out


def _date_in_range(date: str | None, date_range: str) -> bool:
    if not date:
        return False
    start, end = date_range.split(":", 1)
    return start <= date[:10] <= end


def _date_sort_value(date: str | None) -> float:
    if not date:
        return 0.0
    try:
        dt = datetime.fromisoformat(date[:10])
    except ValueError:
        return 0.0
    return dt.year + (dt.timetuple().tm_yday / 366.0)


def _wants_recency(task: str) -> bool:
    return bool(re.search(r"\b(recent|latest|current|newest|prefer recent)\b", task, re.I))
