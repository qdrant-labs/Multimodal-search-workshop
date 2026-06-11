# WORKSHOP EXERCISE: Completed implementation (independent design)
"""
Earnings Call MCP Server.

Four MCP tools that let Claude query the Qdrant collection of earnings call
transcripts. Design notes:

* search_earnings runs a HYBRID query: the Gemini query embedding is matched
  against BOTH named vectors (`text` and `audio` — they live in the same
  multimodal space) and the two rankings are fused with Reciprocal Rank
  Fusion. A chunk that reads well *and* sounds on-topic outranks one that
  only matches in a single modality.
* Recency boosting is a plain client-side rerank with exponential half-life
  decay — easy to read, easy to tune, no server-side scoring DSL involved.
* recommend_similar uses Qdrant's native Recommend API with the seed point
  ID as a positive example; Qdrant excludes the example from its own results.
* get_audio_clip falls back to slicing the full call recording with ffmpeg
  when no pre-sliced clip exists.

How to run:
    python mcp_server/server.py

Register with Claude Desktop / Claude Code:
    python cli/setup_mcp.py install
"""

import base64
import functools
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).parent.parent))
from mcp_server.embeddings import embed_query

# ── Configuration (from .env) ─────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH = os.getenv("QDRANT_PATH") or None
COLLECTION = os.getenv("COLLECTION_NAME", "earnings_calls")

REPO_ROOT = Path(__file__).parent.parent
CLIPS_DIR = Path(os.getenv("CLIPS_DIR", REPO_ROOT / "data" / "audio_clips"))
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", REPO_ROOT / "data" / "audio"))
NEWS_CACHE_DIR = REPO_ROOT / "data" / "asknews_cache"

TOP_K = 5
# Recency rerank: bonus weight and the age (days) at which the bonus halves.
RECENCY_WEIGHT = 0.25
RECENCY_HALF_LIFE_DAYS = 180.0

_client: QdrantClient | None = None


def _qdrant() -> QdrantClient:
    """Lazily create the Qdrant client so importing this module stays cheap."""
    global _client
    if _client is None:
        if QDRANT_URL:
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            _client = QdrantClient(path=QDRANT_PATH or str(REPO_ROOT / "data" / "qdrant_storage"))
    return _client


mcp = FastMCP("earnings-call-server")


# ── Shared helpers ────────────────────────────────────────────────────────────


def fail_soft(fn: Callable) -> Callable:
    """Convert exceptions into an {"error": ...} payload matching the
    function's return type, so MCP clients always get structured output."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # surfaced to the LLM, never raised
            import typing

            error = {"error": f"{type(exc).__name__}: {exc}"}
            annotation = fn.__annotations__.get("return")
            returns_list = typing.get_origin(annotation) is list or (
                isinstance(annotation, str) and annotation.startswith("list")
            )
            return [error] if returns_list else error

    return wrapper


def _conditions(ticker: str | None, date_range: str | None) -> list[models.Condition]:
    """Translate the optional tool args into Qdrant filter conditions."""
    conds: list[models.Condition] = []
    if ticker:
        conds.append(
            models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.strip().upper()))
        )
    if date_range:
        try:
            lo, hi = (part.strip() for part in date_range.split(":", 1))
        except ValueError as exc:
            raise ValueError(f"date_range must look like YYYY-MM-DD:YYYY-MM-DD, got {date_range!r}") from exc
        # `date` is DATETIME-indexed, so use a datetime range (not numeric).
        conds.append(
            models.FieldCondition(
                key="date",
                range=models.DatetimeRange(
                    gte=datetime.fromisoformat(lo) if lo else None,
                    lte=datetime.fromisoformat(hi) if hi else None,
                ),
            )
        )
    return conds


def _fetch_point(point_id: str, with_vectors: bool = False) -> models.Record | None:
    records = _qdrant().retrieve(
        collection_name=COLLECTION,
        ids=[point_id],
        with_payload=True,
        with_vectors=with_vectors,
    )
    return records[0] if records else None


def _chunk_dict(point: Any, extra_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    payload = point.payload or {}
    base = {
        "point_id": str(point.id),
        "ticker": payload.get("ticker"),
        "company": payload.get("company"),
        "quarter": payload.get("quarter"),
        "year": payload.get("year"),
        "chunk_text": payload.get("chunk_text"),
        "speaker": payload.get("speaker"),
        "start_time": payload.get("start_time"),
        "score": getattr(point, "score", None),
    }
    for key in extra_keys:
        base[key] = payload.get(key)
    return base


def _recency_bonus(date_str: str | None, now: datetime) -> float:
    """Exponential half-life bonus in [0, 1]: 1.0 today, 0.5 after one
    half-life, 0.25 after two, ..."""
    if not date_str:
        return 0.0
    try:
        called = datetime.fromisoformat(str(date_str)).replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.0
    age_days = max((now - called).total_seconds() / 86400.0, 0.0)
    return math.pow(0.5, age_days / RECENCY_HALF_LIFE_DAYS)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: search_earnings
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool()
@fail_soft
def search_earnings(
    query: str,
    ticker: str | None = None,
    date_range: str | None = None,
    boost_recency: bool = False,
) -> list[dict[str, Any]]:
    """
    Semantic search over earnings call transcripts stored in Qdrant.

    The query is embedded once with Gemini and matched against BOTH named
    vectors of every chunk — `text` (transcript embedding) and `audio`
    (embedding of the raw 30s clip, same multimodal space) — and the two
    rankings are merged with Reciprocal Rank Fusion.

    Args:
        query:         Natural-language question, e.g. "data center demand outlook"
        ticker:        Optional stock ticker to restrict results, e.g. "NVDA"
        date_range:    Optional ISO date range "YYYY-MM-DD:YYYY-MM-DD"
        boost_recency: When True, recent calls are boosted by an exponential
                       half-life bonus (client-side rerank, fully transparent).

    Returns:
        List of matching transcript chunks with metadata and relevance scores.
    """
    vector = embed_query(query)
    conds = _conditions(ticker, date_range)
    qfilter = models.Filter(must=conds) if conds else None

    # Pull more candidates than we return; the fusion + optional rerank
    # decide the final order.
    pool = TOP_K * 6 if boost_recency else TOP_K * 3

    hybrid = _qdrant().query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=vector, using="text", filter=qfilter, limit=pool),
            models.Prefetch(query=vector, using="audio", filter=qfilter, limit=pool),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=pool,
        with_payload=True,
    )
    hits = list(hybrid.points)

    if boost_recency:
        now = datetime.now(timezone.utc)
        hits.sort(
            key=lambda p: (p.score or 0.0)
            + RECENCY_WEIGHT * _recency_bonus((p.payload or {}).get("date"), now),
            reverse=True,
        )

    return [_chunk_dict(p, extra_keys=("date",)) for p in hits[:TOP_K]]


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: get_audio_clip
# ─────────────────────────────────────────────────────────────────────────────


def _slice_with_ffmpeg(source: Path, dest: Path, start: float, end: float) -> bool:
    """Cut [start, end] out of the full call recording. Uses stream copy, so
    no re-encode and near-instant."""
    try:
        import static_ffmpeg.run

        ffmpeg, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    except Exception:
        ffmpeg = "ffmpeg"  # hope it's on PATH

    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [ffmpeg, "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
         "-i", str(source), "-acodec", "copy", str(dest)],
        capture_output=True,
        timeout=60,
    )
    return proc.returncode == 0 and dest.exists()


@mcp.tool()
@fail_soft
def get_audio_clip(point_id: str) -> dict[str, Any]:
    """
    Retrieve a base64-encoded audio clip for a specific transcript chunk.

    Serves the pre-sliced data/audio_clips/{point_id}.mp3 when present;
    otherwise cuts the segment out of the full call recording with ffmpeg
    and caches it for next time.

    Args:
        point_id: UUID of the Qdrant point returned by search_earnings.

    Returns:
        Dict with keys: point_id, ticker, audio_base64, start_time,
        end_time, format.  On error, returns {"error": "<message>"}.
    """
    point = _fetch_point(point_id)
    if point is None:
        return {"error": f"Point {point_id} not found in collection {COLLECTION!r}"}

    payload = point.payload or {}
    start = float(payload.get("start_time") or 0.0)
    end = float(payload.get("end_time") or 0.0)

    clip_path = CLIPS_DIR / f"{point_id}.mp3"
    if not clip_path.exists():
        source = AUDIO_DIR / str(payload.get("audio_file") or "")
        if not (payload.get("audio_file") and source.exists() and end > start):
            return {
                "error": (
                    f"No clip at {clip_path} and no source audio to slice from. "
                    "Run the ingestion pipeline to generate audio clips."
                )
            }
        if not _slice_with_ffmpeg(source, clip_path, start, end):
            return {"error": f"ffmpeg failed to slice {source.name} [{start:.1f}s-{end:.1f}s]"}

    return {
        "point_id": point_id,
        "ticker": payload.get("ticker"),
        "audio_base64": base64.b64encode(clip_path.read_bytes()).decode("ascii"),
        "start_time": start,
        "end_time": end,
        "format": "mp3",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: get_news_context
# ─────────────────────────────────────────────────────────────────────────────

FIRECRAWL_SNAPSHOT_CHARS = 1200
FIRECRAWL_MAX_SNAPSHOTS = 3  # per news lookup, to keep latency bounded


def _firecrawl_snapshot(url: str) -> str | None:
    """Fetch a clean-markdown snapshot of a page via the Firecrawl API.

    Used as a fallback when a news article arrives without a summary, so the
    LLM still gets real page content instead of a bare link.
    """
    api_key = os.getenv("FIRECRAWL_API_KEY", "")
    if not api_key:
        return None
    try:
        import httpx

        resp = httpx.post(
            "https://api.firecrawl.dev/v2/scrape",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            timeout=20.0,
        )
        if resp.status_code != 200:
            return None
        markdown = (resp.json().get("data") or {}).get("markdown") or ""
        markdown = markdown.strip()
        return markdown[:FIRECRAWL_SNAPSHOT_CHARS] or None
    except Exception:
        return None


@mcp.tool()
@fail_soft
def get_news_context(point_id: str) -> dict[str, Any]:
    """
    Return AskNews articles relevant to the earnings call chunk.

    Reads the pre-fetched cache under data/asknews_cache/ first; on a miss it
    queries the AskNews search API for coverage in a ±7 day window around the
    call date and writes the result back to the cache.

    Args:
        point_id: UUID of the Qdrant point returned by search_earnings.

    Returns:
        Dict with keys: ticker, date, articles (list of article dicts).
        On error, returns {"error": "<message>"}.
    """
    point = _fetch_point(point_id)
    if point is None:
        return {"error": f"Point {point_id} not found in collection {COLLECTION!r}"}

    payload = point.payload or {}
    ticker = payload.get("ticker", "")
    date = payload.get("date", "")

    # Cache layout from the ingest pipeline: one file per chunk, with a
    # per-call file as a coarser fallback.
    for candidate in (
        NEWS_CACHE_DIR / f"{ticker}_{date}_{point_id}.json",
        NEWS_CACHE_DIR / f"{ticker}_{date}.json",
    ):
        if candidate.exists():
            return json.loads(candidate.read_text())

    api_key = os.getenv("ASKNEWS_API_KEY", "")
    if not (api_key and date):
        return {
            "ticker": ticker,
            "date": date,
            "articles": [],
            "note": "No cached news for this chunk and no ASKNEWS_API_KEY for a live lookup.",
        }

    from asknews_sdk import AskNewsSDK  # deferred: only needed on cache miss

    call_day = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    window = (call_day - timedelta(days=7), call_day + timedelta(days=7))

    sdk = AskNewsSDK(api_key=api_key)
    found = sdk.news.search_news(
        query=f"{payload.get('company', ticker)} {ticker} earnings "
        f"{payload.get('quarter', '')} {payload.get('year', '')}",
        n_articles=10,
        method="kw",
        historical=True,
        start_timestamp=int(window[0].timestamp()),
        end_timestamp=int(window[1].timestamp()),
        return_type="dicts",
    )

    result = {
        "ticker": ticker,
        "date": date,
        "window": f"{window[0].date()} → {window[1].date()}",
        "articles": [
            {
                "title": art.eng_title or art.title,
                "summary": art.summary,
                "sentiment": art.sentiment,
                "source": art.source_id,
                "language": art.language,
                "url": str(art.article_url),
                "published_at": str(art.pub_date),
            }
            for art in (found.as_dicts or [])
        ],
    }

    # Articles that arrive without a summary are just bare links — pull a
    # clean page snapshot for them via Firecrawl (bounded, best-effort).
    snapshots_left = FIRECRAWL_MAX_SNAPSHOTS
    for article in result["articles"]:
        if snapshots_left == 0:
            break
        if article.get("summary") or not article.get("url"):
            continue
        snapshot = _firecrawl_snapshot(str(article["url"]))
        if snapshot:
            article["content_snapshot"] = snapshot
        snapshots_left -= 1

    NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (NEWS_CACHE_DIR / f"{ticker}_{date}_{point_id}.json").write_text(
        json.dumps(result, indent=2)
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3b: get_news_graph
# ─────────────────────────────────────────────────────────────────────────────

GRAPH_TOP_NODES = 25
GRAPH_TOP_EDGES = 40
# Empirically the graph endpoint reports "No sources recovered" for narrow
# (±7 day) historical windows and for verbose / ticker-laden keyword queries;
# a 30-day window with a plain query builds reliably.
GRAPH_WINDOW_DAYS = 30


def _compact_graph(raw: dict[str, Any], query: str, ticker: str | None) -> dict[str, Any]:
    """Reduce a raw AskNews graph response to the top nodes/edges by count.

    The live API names fields `article_count` / `from_id` / `to_id`; the docs
    show `count` / `from` / `to` — accept both.
    """

    def _count(item: dict[str, Any]) -> int:
        return item.get("count") or item.get("article_count") or 0

    full = raw.get("full_graph") or {}
    nodes = sorted(full.get("nodes") or [], key=_count, reverse=True)
    edges = sorted(full.get("edges") or [], key=_count, reverse=True)
    return {
        "query": query,
        # The HyDE-expanded query that was actually sent to build_graph
        # (stashed on the cached raw response for transparency). Falls back
        # to the raw query for graphs built before HyDE existed.
        "graph_query": raw.get("_graph_query") or query,
        "ticker": ticker,
        "nodes": [
            {"id": n.get("id"), "type": n.get("type"), "count": _count(n)}
            for n in nodes[:GRAPH_TOP_NODES]
        ],
        "edges": [
            {
                "from": e.get("from") or e.get("from_id"),
                "label": e.get("label"),
                "to": e.get("to") or e.get("to_id"),
            }
            for e in edges[:GRAPH_TOP_EDGES]
        ],
        "visualize_url": raw.get("visualize_url"),
        "triples_url": raw.get("triples_url"),
    }


def _ticker_company(ticker: str) -> str | None:
    """Look up the company name for a ticker from any indexed point."""
    sample, _ = _qdrant().scroll(
        collection_name=COLLECTION,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker))]
        ),
        limit=1,
        with_payload=["company"],
    )
    return (sample[0].payload or {}).get("company") if sample else None


# HyDE (Hypothetical Document Embedding) query expansion for the graph.
# The user's query at search time is often vague or anaphoric ("how did this
# compare to the previous quarter?", "what about that segment?"). The AI
# summary shown above the graph is concrete grounding context — feed both to
# Gemini and have it rewrite the vague query into ONE entity-anchored,
# natural-language news-search query that AskNews' graph endpoint can resolve.
HYDE_MODEL = "models/gemini-3.1-flash-lite"


def _hyde_graph_query(query: str, ticker: str | None, context: str) -> str | None:
    """Rewrite a (possibly vague) query into a concrete graph-search query.

    Resolves anaphora ("this/that/the call/previous quarter") into the
    concrete companies, people, products, events and timeframe implied by the
    provided context (the AI summary). Stays grounded ONLY in the context — no
    invented facts. Keeps the output keyword-rich but readable (a sentence or
    two), NOT a comma-stuffed entity dump, to respect the AskNews 400002
    "No sources recovered" constraint. Returns a single plain-text line, or
    None when Gemini is unavailable/fails so the caller can fall back.
    """
    context = (context or "").strip()
    if not (context and os.getenv("GEMINI_API_KEY")):
        return None
    try:
        from google import genai

        company = _ticker_company(ticker) if ticker else None
        focus = f"{company} ({ticker})" if company else (ticker or "the company in the context")
        prompt = (
            "You rewrite a user's news-search query so a knowledge-graph search "
            "engine can find relevant articles.\n\n"
            "CONTEXT (an AI-generated summary of earnings-call search results — "
            "your ONLY source of truth):\n"
            f"{context}\n\n"
            f'USER QUERY: "{query}"\n'
            f"PRIMARY SUBJECT: {focus}\n\n"
            "Rewrite the USER QUERY into ONE concise, natural-language news-search "
            "query that:\n"
            "- Resolves every vague or relative reference (this, that, it, the "
            "call, the quarter, previous/next quarter, the segment, prev call) "
            "into the concrete company names, people, products, events and an "
            "explicit timeframe implied by the CONTEXT.\n"
            "- Is grounded ONLY in the CONTEXT. Do not invent facts, numbers, "
            "tickers, or events that are not supported by it.\n"
            "- Reads like a natural sentence or two and is keyword-rich, but is "
            "NOT a comma-separated dump of entities (that breaks the search).\n"
            "- Names the primary subject company explicitly.\n\n"
            "Return ONLY the rewritten query as a single line of plain text, no "
            "quotes, no preamble, no markdown."
        )
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(model=HYDE_MODEL, contents=prompt)
        text = (response.text or "").strip()
        if not text:
            return None
        # Collapse to a single line; strip any stray surrounding quotes.
        line = " ".join(text.splitlines()).strip().strip('"').strip()
        return line or None
    except Exception:
        return None


@mcp.tool()
@fail_soft
def get_news_graph(
    query: str,
    ticker: str | None = None,
    date_range: str | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    """
    Build an AskNews knowledge graph of news around a search query.

    Calls the AskNews graph endpoint with the natural-language query (plus the
    company name when a ticker is given). News is date-filtered by date_range
    when provided, otherwise the last GRAPH_WINDOW_DAYS days. The full raw
    response is cached under data/asknews_cache/graph_*.json and reused on
    subsequent calls (the graph endpoint consumes AskNews credits).

    When `context` is supplied (e.g. the AI summary rendered above the graph),
    a HyDE-style query expansion (Gemini) rewrites a vague/anaphoric `query`
    into a concrete, entity-anchored news-search query grounded in that
    context before calling build_graph. The expanded query is returned as
    `graph_query` for transparency; on any failure it falls back to `query`.

    Args:
        query:      Natural-language search query, e.g. "data center demand".
        ticker:     Optional stock ticker to focus the graph, e.g. "NVDA".
        date_range: Optional ISO date range "YYYY-MM-DD:YYYY-MM-DD".
        context:    Optional grounding text (the AI summary) used to resolve
                    vague references in `query` via HyDE expansion.

    Returns:
        Dict with keys: query (raw), graph_query (HyDE-expanded query actually
        used), ticker, nodes (top entities, [{id, type, count}]), edges (top
        relationships, [{from, label, to}]), visualize_url (hosted interactive
        visualization), triples_url. On error, returns {"error": "<message>"}.
    """
    query = query.strip()
    if not query:
        return {"error": "query is required"}
    ticker = ticker.strip().upper() if ticker else None
    context = (context or "").strip()

    # Different summaries (contexts) should resolve to different graph queries,
    # so fold a short hash of the context into the cache key.
    ctx_hash = hashlib.sha256(context.encode()).hexdigest()[:8] if context else ""
    cache_key = hashlib.sha256(
        f"{query}|{ticker or ''}|{date_range or ''}|{ctx_hash}".encode()
    ).hexdigest()[:16]
    cache_path = NEWS_CACHE_DIR / f"graph_{cache_key}.json"
    if cache_path.exists():
        return _compact_graph(json.loads(cache_path.read_text()), query, ticker)

    api_key = os.getenv("ASKNEWS_API_KEY", "")
    if not api_key:
        return {"error": "No cached graph for this query and no ASKNEWS_API_KEY for a live build."}

    if date_range:
        try:
            lo, hi = (part.strip() for part in date_range.split(":", 1))
        except ValueError as exc:
            raise ValueError(
                f"date_range must look like YYYY-MM-DD:YYYY-MM-DD, got {date_range!r}"
            ) from exc
        start = datetime.fromisoformat(lo).replace(tzinfo=timezone.utc) if lo else None
        end = datetime.fromisoformat(hi).replace(tzinfo=timezone.utc) if hi else None
    else:
        now = datetime.now(timezone.utc)
        start, end = now - timedelta(days=GRAPH_WINDOW_DAYS), now

    filter_params: dict[str, Any] = {"historical": True}
    if start:
        filter_params["start_timestamp"] = int(start.timestamp())
    if end:
        filter_params["end_timestamp"] = int(end.timestamp())

    # HyDE expansion when we have grounding context; otherwise keep the
    # historical behaviour of lightly anchoring the query with the company
    # name. Either way `graph_query` is what actually hits build_graph.
    graph_query = _hyde_graph_query(query, ticker, context) if context else None
    if not graph_query:
        graph_query = f"{query} {_ticker_company(ticker) or ticker}" if ticker else query

    from asknews_sdk import AskNewsSDK  # deferred: only needed on cache miss

    sdk = AskNewsSDK(api_key=api_key)
    graph = sdk.news.build_graph(
        query=graph_query,
        filter_params=filter_params,
        visualize_with="cosmograph.app",
    )

    raw = graph.model_dump(mode="json")
    # Stash both queries on the cached raw response for transparency.
    raw["_query"] = query
    raw["_graph_query"] = graph_query
    NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(raw, indent=2))
    return _compact_graph(raw, query, ticker)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: recommend_similar
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool()
@fail_soft
def recommend_similar(point_id: str) -> list[dict[str, Any]]:
    """
    Find transcript chunks that are semantically similar to a given chunk.

    Uses Qdrant's native Recommend API: the seed point ID is passed as a
    positive example, Qdrant looks up its stored `text` vector server-side
    and excludes the example from the results automatically — no manual
    vector round-trip needed.

    Args:
        point_id: UUID of the Qdrant point to use as the reference.

    Returns:
        List of up to 5 similar chunks with point_id, ticker, chunk_text,
        score, quarter, year.  On error, returns [{"error": "<message>"}].
    """
    if _fetch_point(point_id) is None:
        return [{"error": f"Point {point_id} not found in collection {COLLECTION!r}"}]

    similar = _qdrant().query_points(
        collection_name=COLLECTION,
        query=models.RecommendQuery(
            recommend=models.RecommendInput(positive=[point_id])
        ),
        using="text",
        limit=TOP_K,
        with_payload=True,
    )

    return [_chunk_dict(p) for p in similar.points]


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5: list_tickers
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
@fail_soft
def list_tickers() -> dict[str, Any]:
    """
    Show which tickers (earnings calls) are currently indexed in Qdrant,
    with chunk counts and call metadata, plus any indexing jobs in flight.

    Returns:
        Dict with:
            indexed:  list of {ticker, company, quarter, year, date, chunks}
            jobs:     active indexing jobs started via index_earnings_call
            total_chunks: total points in the collection
    """
    client = _qdrant()

    facet = client.facet(collection_name=COLLECTION, key="ticker", limit=100)

    indexed: list[dict[str, Any]] = []
    for hit in facet.hits:
        ticker = str(hit.value)
        # Grab one point per ticker for the call-level metadata.
        sample, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker))]
            ),
            limit=1,
            with_payload=True,
        )
        payload = (sample[0].payload or {}) if sample else {}
        indexed.append(
            {
                "ticker": ticker,
                "company": payload.get("company"),
                "quarter": payload.get("quarter"),
                "year": payload.get("year"),
                "date": payload.get("date"),
                "chunks": hit.count,
            }
        )

    from mcp_server import jobs

    active = [
        {
            "job_id": j["job_id"],
            "ticker": j["ticker"],
            "quarter": j["quarter"],
            "year": j["year"],
            "status": j["status"],
            "stage": j.get("stage"),
            "percent": j.get("percent", 0),
        }
        for j in jobs.load_jobs()
        if j["status"] in ("pending", "running")
    ]

    return {
        "indexed": sorted(indexed, key=lambda c: c["ticker"]),
        "jobs": active,
        "total_chunks": sum(c["chunks"] for c in indexed),
        "note": "Use index_earnings_call(...) to start an indexing job for a new call.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 6: index_earnings_call
# ─────────────────────────────────────────────────────────────────────────────


def _write_access() -> bool:
    """Best-effort check whether the configured Qdrant credentials can write.

    Cloud keys are JWTs whose payload carries an `access` claim ('r' =
    read-only). Local/path mode is always writable.
    """
    if not QDRANT_URL:
        return True
    key = QDRANT_API_KEY or ""
    try:
        claims_b64 = key.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(claims_b64 + "=" * (-len(claims_b64) % 4)))
        return claims.get("access") != "r"
    except Exception:
        return True  # not a JWT — assume a full-access key


def _firecrawl_search(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Web search via the Firecrawl API. Returns [{url, title, description}]."""
    api_key = os.getenv("FIRECRAWL_API_KEY", "")
    if not api_key:
        return []
    import httpx

    resp = httpx.post(
        "https://api.firecrawl.dev/v2/search",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": query, "limit": limit},
        timeout=30.0,
    )
    if resp.status_code != 200:
        return []
    data = resp.json().get("data") or {}
    hits = data.get("web") or (data if isinstance(data, list) else [])
    return [
        {
            "url": h.get("url", ""),
            "title": h.get("title", ""),
            "description": h.get("description", ""),
        }
        for h in hits
        if h.get("url")
    ]


_QUARTER_RE = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:FY\s?)?(20\d{2})\b")


def _guess_quarter_year(text: str) -> tuple[str | None, int | None]:
    quarter = None
    year = None
    if m := _QUARTER_RE.search(text):
        quarter = f"Q{m.group(1)}"
    if m := _YEAR_RE.search(text):
        year = int(m.group(1))
    return quarter, year


def _gemini_normalize_candidates(
    ticker: str,
    candidates: list[dict[str, Any]],
    news_hints: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Let Gemini turn raw search hits into clean, queue-ready candidates.

    The regex pass over titles is noisy (live-stream clips, wrong FY labels,
    duplicate uploads). Gemini cross-references the news hints (which carry
    exact publication dates of earnings coverage) and returns one structured
    entry per actual call. Returns None when Gemini is unavailable/fails so
    the caller can keep the regex-based fallback.
    """
    if not os.getenv("GEMINI_API_KEY"):
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        prompt = (
            f"You are cleaning up earnings-call search results for ticker {ticker}.\n"
            "Below are RAW_VIDEOS (YouTube search hits) and NEWS_HINTS (dated "
            "earnings coverage; transcript articles are usually published on or "
            "right after the call date).\n\n"
            "Return a JSON array where each element represents ONE distinct "
            "earnings call, with keys:\n"
            "  youtube_url (pick the best full-recording video for that call),\n"
            "  title, company (full company name for the ticker),\n"
            "  quarter ('Q1'..'Q4', fiscal label as the company uses it),\n"
            "  year (int), date ('YYYY-MM-DD' of the call if derivable from the "
            "news hints, else null), confidence (0-1).\n"
            "Exclude videos that are not full earnings-call recordings (shorts, "
            "reaction clips, previews). Deduplicate multiple uploads of the same "
            "call. Return ONLY the JSON array.\n\n"
            f"RAW_VIDEOS:\n{json.dumps(candidates, indent=1)}\n\n"
            f"NEWS_HINTS:\n{json.dumps(news_hints, indent=1)}"
        )
        response = client.models.generate_content(
            model="models/gemini-3.1-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        cleaned = json.loads(response.text or "[]")
        if not isinstance(cleaned, list):
            return None
        return [
            {
                "youtube_url": c.get("youtube_url"),
                "title": c.get("title"),
                "company": c.get("company"),
                "quarter": c.get("quarter"),
                "year": c.get("year"),
                "date": c.get("date"),
                "confidence": c.get("confidence"),
            }
            for c in cleaned
            if c.get("youtube_url")
        ]
    except Exception:
        return None


@mcp.tool()
@fail_soft
def discover_earnings_calls(ticker: str, company: str = "", years_back: int = 4) -> dict[str, Any]:
    """
    Discover available earnings calls for a ticker from the last N years,
    using Firecrawl web search (YouTube recordings + IR/report pages) and
    AskNews coverage (to pin down call dates).

    Candidates with a youtube_url can be fed straight into
    index_earnings_call(...) to start an indexing job.

    Args:
        ticker:     Stock ticker, e.g. "MSFT".
        company:    Optional company name to sharpen the search.
        years_back: How many years of history to look for (default 4).

    Returns:
        Dict with `candidates` (recordings, queue-ready), `reports`
        (10-Q/10-K/IR pages), and `news_hints` (dated earnings coverage).
    """
    ticker = ticker.strip().upper()
    name = company or ticker
    this_year = datetime.now(timezone.utc).year
    year_lo = this_year - years_back

    already = {
        (c["ticker"], c.get("quarter"), c.get("year"))
        for c in list_tickers().get("indexed", [])
    }

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for y in range(this_year, year_lo - 1, -1):
        for hit in _firecrawl_search(
            f"{name} {ticker} earnings call {y} site:youtube.com", limit=5
        ):
            if hit["url"] in seen:
                continue
            seen.add(hit["url"])
            blob = f"{hit['title']} {hit['description']}"
            quarter, year = _guess_quarter_year(blob)
            candidates.append(
                {
                    "youtube_url": hit["url"],
                    "title": hit["title"],
                    "quarter": quarter,
                    "year": year or y,
                    "already_indexed": (ticker, quarter, year or y) in already,
                }
            )

    reports = _firecrawl_search(
        f"{name} {ticker} quarterly report 10-Q 10-K investor relations", limit=5
    )

    news_hints: list[dict[str, Any]] = []
    asknews_key = os.getenv("ASKNEWS_API_KEY", "")
    if asknews_key:
        from asknews_sdk import AskNewsSDK

        sdk = AskNewsSDK(api_key=asknews_key)
        now = datetime.now(timezone.utc)
        # AskNews historical search caps a query at 160 days — walk the
        # lookback window in ~150-day slices.
        slice_days = 150
        window_end = now
        window_start_limit = now.replace(year=now.year - years_back)
        seen_news: set[str] = set()
        while window_end > window_start_limit:
            window_start = max(window_end - timedelta(days=slice_days), window_start_limit)
            try:
                found = sdk.news.search_news(
                    query=f"{name} {ticker} quarterly earnings results call",
                    n_articles=3,
                    method="kw",
                    historical=True,
                    start_timestamp=int(window_start.timestamp()),
                    end_timestamp=int(window_end.timestamp()),
                    return_type="dicts",
                )
            except Exception:
                found = None  # one bad slice shouldn't kill discovery
            for art in (found.as_dicts or []) if found else []:
                url = str(art.article_url)
                if url in seen_news:
                    continue
                seen_news.add(url)
                quarter, year = _guess_quarter_year(f"{art.eng_title or art.title} {art.summary}")
                news_hints.append(
                    {
                        "title": art.eng_title or art.title,
                        "published_at": str(art.pub_date),
                        "quarter": quarter,
                        "year": year,
                        "url": url,
                    }
                )
            window_end = window_start
        news_hints.sort(key=lambda h: h["published_at"], reverse=True)

    # Gemini pass: dedupe uploads, drop non-call clips, attach call dates
    # from the news hints. Falls back to the regex guesses when unavailable.
    normalized = _gemini_normalize_candidates(ticker, candidates, news_hints)
    if normalized:
        for c in normalized:
            c["already_indexed"] = (ticker, c.get("quarter"), c.get("year")) in already
        candidates = normalized

    return {
        "ticker": ticker,
        "window": f"{year_lo} → {this_year}",
        "candidates": candidates,
        "reports": reports,
        "news_hints": news_hints,
        "note": "Pass a candidate's youtube_url (+ quarter/year/date) to index_earnings_call() to start a job.",
    }


@mcp.tool()
@fail_soft
def index_earnings_call(
    youtube_url: str,
    ticker: str,
    company: str,
    quarter: str,
    year: int,
    date: str,
) -> dict[str, Any]:
    """
    Start a background indexing JOB for a new earnings call.

    The job runs the full pipeline against the local writable Qdrant with
    per-stage progress you can poll via get_indexing_jobs():

        download    yt-dlp pulls the call audio from YouTube
        transcribe  whisper + pyannote diarization when available,
                    else Gemini flash-lite per 30s clip
        embed       ingest/03 pipeline: gemini-embedding-2 text+audio
                    vectors → Qdrant upsert

    Args:
        youtube_url: Full YouTube URL of the earnings call recording.
        ticker:      Stock ticker, e.g. "MSFT".
        company:     Company name, e.g. "Microsoft Corporation".
        quarter:     Fiscal quarter label, e.g. "Q2".
        year:        Fiscal year, e.g. 2025.
        date:        Call date as YYYY-MM-DD.

    Returns:
        The created job record (job_id, status, stage, percent, ...).
    """
    ticker = ticker.strip().upper()
    datetime.strptime(date, "%Y-%m-%d")  # validates format
    if "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
        return {"error": f"youtube_url does not look like a YouTube link: {youtube_url!r}"}
    if not _write_access():
        return {"error": "Configured Qdrant target is read-only — cannot index new calls into it."}
    if not os.getenv("GEMINI_API_KEY"):
        return {"error": "GEMINI_API_KEY missing — needed for transcription and embeddings."}

    # Already indexed?  Only ticker and year are keyword/integer-indexed on
    # the shared cluster, so filter on those and confirm quarter client-side.
    existing, _ = _qdrant().scroll(
        collection_name=COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker)),
                models.FieldCondition(key="year", match=models.MatchValue(value=year)),
            ]
        ),
        limit=16,
        with_payload=["quarter"],
    )
    if any((p.payload or {}).get("quarter") == quarter for p in existing):
        return {
            "status": "already_indexed",
            "detail": f"{ticker} {quarter} {year} is already in the collection.",
        }

    from mcp_server import jobs

    return jobs.start_job(
        youtube_url=youtube_url,
        ticker=ticker,
        company=company,
        quarter=quarter,
        year=year,
        date=date,
    )


@mcp.tool()
@fail_soft
def get_indexing_jobs(job_id: str | None = None) -> list[dict[str, Any]]:
    """
    Report indexing jobs and their progress.

    Each job shows: status (pending/running/done/failed/cancelled), the
    current stage (download/chunk/transcribe/embed), overall percent,
    per-stage done/total counters, and the last log lines.

    Args:
        job_id: Optional — return just this job; otherwise all jobs,
                newest first.
    """
    from mcp_server import jobs

    if job_id:
        job = jobs.load_job(job_id)
        return [job] if job else [{"error": f"Job {job_id} not found"}]
    return jobs.load_jobs()


@mcp.tool()
@fail_soft
def cancel_indexing_job(job_id: str) -> dict[str, Any]:
    """
    Request cancellation of a running indexing job. The worker stops at the
    next chunk boundary; completed work (downloads, clips, cached vectors)
    is kept so a rerun resumes cheaply.

    Args:
        job_id: ID returned by index_earnings_call / get_indexing_jobs.
    """
    from mcp_server import jobs

    job = jobs.cancel_job(job_id)
    return job if job else {"error": f"Job {job_id} not found"}


# ─────────────────────────────────────────────────────────────────────────────
# Tools 7-9: collection lifecycle (clone status / clone / delete)
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool()
@fail_soft
def get_clone_status() -> dict[str, Any]:
    """
    Report the state of the working collection vs. the cloud source.

    Shows whether the collection exists on the configured (local) Qdrant,
    how many points it holds, how many the read-only workshop cloud source
    holds, and whether the two are in sync — so you can decide to clone,
    re-clone, or delete and reindex.

    Returns:
        Dict with: target_url, collection, exists, points, writable,
        cloud_url, cloud_points, in_sync.
    """
    client = _qdrant()
    exists = client.collection_exists(COLLECTION)
    points = (client.get_collection(COLLECTION).points_count or 0) if exists else 0

    cloud_url = os.getenv("CLOUD_QDRANT_URL", "")
    cloud_points: int | None = None
    cloud_error: str | None = None
    if cloud_url:
        try:
            cloud = QdrantClient(
                url=cloud_url, api_key=os.getenv("CLOUD_QDRANT_API_KEY") or None, timeout=15
            )
            cloud_points = cloud.get_collection(COLLECTION).points_count or 0
        except Exception as exc:
            cloud_error = str(exc)

    return {
        "target_url": QDRANT_URL or f"path:{QDRANT_PATH}",
        "collection": COLLECTION,
        "exists": exists,
        "points": points,
        "writable": _write_access(),
        "cloud_url": cloud_url or None,
        "cloud_points": cloud_points,
        "cloud_error": cloud_error,
        "in_sync": cloud_points is not None and points == cloud_points,
    }


@mcp.tool()
@fail_soft
def clone_collection(force: bool = False) -> dict[str, Any]:
    """
    Clone (or re-clone) the earnings_calls collection from the read-only
    workshop cloud cluster into the configured writable Qdrant.

    Copies every point — payloads plus both `text` and `audio` named
    vectors — and recreates the payload indexes. Re-cloning DELETES the
    existing local collection first.

    Args:
        force: Must be True to overwrite an existing non-empty collection.

    Returns:
        Dict with cloud_points / local_points on success.
    """
    if not _write_access():
        return {"error": "Configured QDRANT credentials are read-only — cannot clone into this target."}
    if not os.getenv("CLOUD_QDRANT_URL"):
        return {"error": "CLOUD_QDRANT_URL is not set — no source to clone from."}

    client = _qdrant()
    if client.collection_exists(COLLECTION):
        existing = client.get_collection(COLLECTION).points_count or 0
        if existing and not force:
            return {
                "error": (
                    f"Collection {COLLECTION!r} already has {existing} points. "
                    "Pass force=True to delete and re-clone."
                )
            }

    from scripts.migrate_to_local import run_migration

    stats = run_migration()
    return {"status": "cloned", **stats}


@mcp.tool()
@fail_soft
def delete_collection(confirm: bool = False) -> dict[str, Any]:
    """
    Delete the working collection from the configured (local) Qdrant so it
    can be re-cloned or rebuilt from scratch with the ingestion pipeline.

    Args:
        confirm: Must be True — this irreversibly drops the local collection.

    Returns:
        Dict with the deletion status.
    """
    if not confirm:
        return {"error": "Refusing to delete: call with confirm=True."}
    if not _write_access():
        return {"error": "Configured QDRANT credentials are read-only — cannot delete."}

    client = _qdrant()
    if not client.collection_exists(COLLECTION):
        return {"status": "absent", "detail": f"Collection {COLLECTION!r} does not exist."}

    points = client.get_collection(COLLECTION).points_count or 0
    client.delete_collection(COLLECTION)
    return {
        "status": "deleted",
        "detail": f"Dropped {COLLECTION!r} ({points} points). "
        "Use clone_collection() or the ingestion pipeline to rebuild.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
