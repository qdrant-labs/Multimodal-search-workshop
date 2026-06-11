# WORKSHOP EXERCISE: Completed implementation
"""
Earnings Call MCP Server — completed workshop implementation.

Implements the four MCP tools so that Claude can query the Qdrant
collection of earnings call transcripts:

    1. search_earnings    — semantic search (+ optional recency boost, Ex 6)
    2. get_audio_clip     — base64 audio for a chunk (with pydub fallback)
    3. get_news_context   — cached/live AskNews context for a chunk
    4. recommend_similar  — "more like this" via stored text vectors

How to run:
    python mcp_server/server.py

To register with Claude Desktop / Claude Code first run:
    python cli/setup_mcp.py install
"""

import base64
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ── Third-party imports ──────────────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Condition,
    DatetimeExpression,
    DatetimeKeyExpression,
    DatetimeRange,
    DecayParamsExpression,
    ExpDecayExpression,
    FieldCondition,
    Filter,
    FormulaQuery,
    HasIdCondition,
    MatchValue,
    MultExpression,
    Prefetch,
    SumExpression,
)

# Local embedding helper (handles caching for offline mode)
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcp_server.embeddings import embed_query

# ── Configuration (from .env) ─────────────────────────────────────────────────
QDRANT_URL: str | None = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH: str | None = os.getenv("QDRANT_PATH") or None
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "earnings_calls")
CLIPS_DIR: Path = Path(os.getenv("CLIPS_DIR", "./data/audio_clips"))
ASKNEWS_CACHE_DIR: Path = Path("./data/asknews_cache")
AUDIO_DIR: Path = Path(os.getenv("AUDIO_DIR", "./data/audio"))

# Recency boost (Exercise 6): weight of the decay bonus and the half-life
# (in days) after which an older call's bonus halves.
RECENCY_BOOST_WEIGHT = 0.3
RECENCY_HALF_LIFE_DAYS = 180

# ── Qdrant client ────────────────────────────────────────────────────────────
if QDRANT_URL:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    client = QdrantClient(path=QDRANT_PATH or "./data/qdrant_storage")

# ── FastMCP app ───────────────────────────────────────────────────────────────
mcp = FastMCP("earnings-call-server")


def _format_chunk(point: Any) -> dict[str, Any]:
    """Shared result formatting for search-style tools."""
    payload = point.payload or {}
    return {
        "point_id": str(point.id),
        "ticker": payload.get("ticker"),
        "company": payload.get("company"),
        "quarter": payload.get("quarter"),
        "year": payload.get("year"),
        "chunk_text": payload.get("chunk_text"),
        "speaker": payload.get("speaker"),
        "start_time": payload.get("start_time"),
        "score": point.score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: search_earnings
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool()
def search_earnings(
    query: str,
    ticker: str | None = None,
    date_range: str | None = None,
    boost_recency: bool = False,
) -> list[dict[str, Any]]:
    """
    Semantic search over earnings call transcripts stored in Qdrant.

    Args:
        query:         Natural-language question, e.g. "data center demand outlook"
        ticker:        Optional stock ticker to restrict results, e.g. "NVDA"
        date_range:    Optional ISO date range "YYYY-MM-DD:YYYY-MM-DD"
                       e.g. "2023-01-01:2024-01-01"
        boost_recency: When True, rerank by combining semantic similarity with a
                       time-decay bonus so more recent calls surface higher.

    Returns:
        List of matching transcript chunks with metadata and relevance scores.
    """
    try:
        # Step 1: Embed the query text (3072-dim, shared multimodal space).
        query_vector = embed_query(query)

        # Step 2: Build an optional Qdrant filter.
        # `date` is DATETIME-indexed, so use DatetimeRange with ISO strings.
        conditions: list[Condition] = []
        if ticker:
            conditions.append(
                FieldCondition(key="ticker", match=MatchValue(value=ticker.upper()))
            )

        if date_range:
            parts = [p.strip() for p in date_range.split(":")]
            if len(parts) == 2:
                start_date, end_date = parts
                conditions.append(
                    FieldCondition(
                        key="date",
                        range=DatetimeRange(gte=start_date, lte=end_date),
                    )
                )

        qdrant_filter = Filter(must=conditions) if conditions else None

        # Step 3: Run the vector search against the `text` named vector.
        if boost_recency:
            # Exercise 6 — prefetch a wide candidate pool by pure similarity,
            # then rerank with: final = $score + WEIGHT * exp_decay(now - date)
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=Prefetch(
                    query=query_vector,
                    using="text",
                    filter=qdrant_filter,
                    limit=30,
                ),
                query=FormulaQuery(
                    formula=SumExpression(
                        sum=[
                            "$score",
                            MultExpression(
                                mult=[
                                    RECENCY_BOOST_WEIGHT,
                                    ExpDecayExpression(
                                        exp_decay=DecayParamsExpression(
                                            x=DatetimeKeyExpression(datetime_key="date"),
                                            target=DatetimeExpression(datetime=now_iso),
                                            scale=RECENCY_HALF_LIFE_DAYS * 86400,
                                            midpoint=0.5,
                                        )
                                    ),
                                ]
                            ),
                        ]
                    )
                ),
                limit=5,
                with_payload=True,
            )
        else:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                using="text",
                query_filter=qdrant_filter,
                limit=5,
                with_payload=True,
            )

        # Step 4: Format the results.
        return [_format_chunk(r) for r in results.points]

    except Exception as exc:
        return [{"error": str(exc)}]


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: get_audio_clip
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool()
def get_audio_clip(point_id: str) -> dict[str, Any]:
    """
    Retrieve a base64-encoded audio clip for a specific transcript chunk.

    Looks up the point in Qdrant to get its time offsets, then returns the
    pre-sliced audio clip from data/audio_clips/{point_id}.mp3 if it exists.
    Falls back to slicing the full call audio on demand with pydub.

    Args:
        point_id: UUID of the Qdrant point returned by search_earnings.

    Returns:
        Dict with keys: point_id, ticker, audio_base64, start_time,
        end_time, format.  On error, returns {"error": "<message>"}.
    """
    try:
        # Step 1: Retrieve the point from Qdrant
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        if not points:
            return {"error": f"Point {point_id} not found"}

        payload = points[0].payload or {}
        ticker = payload.get("ticker", "")
        start_time = payload.get("start_time", 0.0)
        end_time = payload.get("end_time", 0.0)

        def _clip_response(clip_path: Path) -> dict[str, Any]:
            return {
                "point_id": point_id,
                "ticker": ticker,
                "audio_base64": base64.b64encode(clip_path.read_bytes()).decode(),
                "start_time": start_time,
                "end_time": end_time,
                "format": "mp3",
            }

        # Step 2: Check if a pre-sliced clip exists
        clip_path = CLIPS_DIR / f"{point_id}.mp3"
        if clip_path.exists():
            return _clip_response(clip_path)

        # Step 3: No pre-sliced clip — slice from the full audio on demand
        audio_file = payload.get("audio_file", "")
        full_audio_path = AUDIO_DIR / audio_file
        if audio_file and full_audio_path.exists():
            try:
                from pydub import AudioSegment  # type: ignore

                audio = AudioSegment.from_mp3(str(full_audio_path))
                clip = audio[int(start_time * 1000) : int(end_time * 1000)]
                CLIPS_DIR.mkdir(parents=True, exist_ok=True)
                clip.export(str(clip_path), format="mp3")
                return _clip_response(clip_path)
            except ImportError:
                pass  # pydub not available; fall through to error

        return {
            "error": (
                f"No pre-sliced clip found for {point_id} (expected: {clip_path}). "
                "Run the ingestion pipeline to generate clips, or ensure pydub "
                "is installed for on-the-fly slicing."
            )
        }

    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: get_news_context
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool()
def get_news_context(point_id: str) -> dict[str, Any]:
    """
    Return AskNews articles relevant to the earnings call chunk.

    Looks up the point, extracts ticker and date, then reads the pre-fetched
    AskNews cache file.  Falls back to a live AskNews API call if the cache
    file is missing and credentials are available.

    Args:
        point_id: UUID of the Qdrant point returned by search_earnings.

    Returns:
        Dict with keys: ticker, date, articles (list of article dicts).
        On error, returns {"error": "<message>"}.
    """
    try:
        # Step 1: Retrieve the point from Qdrant
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        if not points:
            return {"error": f"Point {point_id} not found"}

        # Step 2: Extract metadata from the payload
        payload = points[0].payload or {}
        ticker = payload.get("ticker", "")
        company = payload.get("company", "")
        date = payload.get("date", "")  # "YYYY-MM-DD"
        speaker = payload.get("speaker", "")
        chunk_text = payload.get("chunk_text", "")

        # Step 3: Check the disk cache (per-chunk file first, then per-call)
        for cache_name in (f"{ticker}_{date}_{point_id}.json", f"{ticker}_{date}.json"):
            cache_path = ASKNEWS_CACHE_DIR / cache_name
            if cache_path.exists():
                return json.loads(cache_path.read_text())

        # Step 4: Not cached — try a live AskNews call
        asknews_key = os.getenv("ASKNEWS_API_KEY", "")
        if not asknews_key or not date:
            return {
                "ticker": ticker,
                "date": date,
                "articles": [],
                "note": (
                    "No AskNews cache found and ASKNEWS_API_KEY not set. "
                    "Run ingest/04_build_asknews_context.py to pre-populate the cache."
                ),
            }

        from asknews_sdk import AskNewsSDK  # type: ignore

        ask = AskNewsSDK(api_key=asknews_key)
        call_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        window_start = call_dt - timedelta(days=7)
        window_end = call_dt + timedelta(days=7)

        news_query = (
            f"{company} ({ticker}) earnings, market conditions, and company news "
            f"around {date}. Context from the call ({speaker}): {chunk_text[:300]}"
        )
        response = ask.news.search_news(
            query=news_query,
            n_articles=10,
            method="kw",
            historical=True,
            start_timestamp=int(window_start.timestamp()),
            end_timestamp=int(window_end.timestamp()),
            return_type="dicts",
        )

        articles = [
            {
                "title": a.eng_title or a.title,
                "summary": a.summary,
                "sentiment": a.sentiment,
                "source": a.source_id,
                "language": a.language,
                "url": str(a.article_url),
                "published_at": str(a.pub_date),
            }
            for a in response.as_dicts or []
        ]

        result = {
            "ticker": ticker,
            "date": date,
            "window": f"{window_start.date()} → {window_end.date()}",
            "articles": articles,
        }

        # Cache for offline reuse
        ASKNEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = ASKNEWS_CACHE_DIR / f"{ticker}_{date}_{point_id}.json"
        cache_path.write_text(json.dumps(result, indent=2))

        return result

    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: recommend_similar
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool()
def recommend_similar(point_id: str) -> list[dict[str, Any]]:
    """
    Find transcript chunks that are semantically similar to a given chunk.

    Fetches the seed point's stored `text` vector and searches with it,
    excluding the seed itself, to surface related content that may come
    from other quarters or tickers.

    Args:
        point_id: UUID of the Qdrant point to use as the reference.

    Returns:
        List of up to 5 similar chunks with point_id, ticker, chunk_text,
        score, quarter, year.  On error, returns [{"error": "<message>"}].
    """
    try:
        # Step 1: Fetch the seed point's stored vectors.
        pts = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_vectors=True,
        )
        if not pts:
            return [{"error": f"Point {point_id} not found"}]

        seed_vectors = pts[0].vector
        seed_text = (
            seed_vectors["text"] if isinstance(seed_vectors, dict) else seed_vectors
        )

        # Step 2: Search for nearest neighbours, excluding the seed itself.
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=seed_text,
            using="text",
            query_filter=Filter(must_not=[HasIdCondition(has_id=[point_id])]),
            limit=5,
            with_payload=True,
        )

        # Step 3: Format and return results.
        return [
            {
                "point_id": str(r.id),
                "ticker": (r.payload or {}).get("ticker"),
                "chunk_text": (r.payload or {}).get("chunk_text"),
                "score": r.score,
                "quarter": (r.payload or {}).get("quarter"),
                "year": (r.payload or {}).get("year"),
            }
            for r in results.points
        ]

    except Exception as exc:
        return [{"error": str(exc)}]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
