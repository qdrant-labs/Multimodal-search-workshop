"""
Earnings Call MCP Server.

Provides five tools that let Claude query a Qdrant vector database of
earnings call transcripts and audio clips:

    search_earnings    — semantic search with optional filters & recency boost
    get_audio_clip     — retrieve base64-encoded MP3 clip for a chunk
    get_news_context   — fetch AskNews context around an earnings moment
    recommend_similar  — find topically related chunks via stored vectors
    compare_tickers    — side-by-side comparison of multiple companies (new)
    get_sec_filings    — scrape SEC EDGAR 10-Q / 10-K filings (bonus)

Run:
    python mcp_server/server.py

Register with Claude:
    python cli/setup_mcp.py install
"""

import base64
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
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

load_dotenv()

# Ensure ffmpeg is available for on-the-fly audio slicing via pydub
if not shutil.which("ffmpeg"):
    try:
        import static_ffmpeg  # type: ignore
        static_ffmpeg.add_paths()
    except ImportError:
        pass

from mcp_server.embeddings import embed_query  # noqa: E402

# ── Tuning constants ──────────────────────────────────────────────────────────
RECENCY_BOOST_WEIGHT = 0.3
RECENCY_HALF_LIFE_DAYS = 180   # score bonus halves every ~6 months

# ── Configuration ─────────────────────────────────────────────────────────────
QDRANT_URL: str | None = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH: str | None = os.getenv("QDRANT_PATH") or None
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "earnings_calls")
CLIPS_DIR: Path = Path(os.getenv("CLIPS_DIR", "./data/audio_clips"))
AUDIO_DIR: Path = Path(os.getenv("AUDIO_DIR", "./data/audio"))
ASKNEWS_CACHE_DIR: Path = Path("./data/asknews_cache")

# ── Qdrant client ─────────────────────────────────────────────────────────────
if QDRANT_URL:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    client = QdrantClient(path=QDRANT_PATH or "./data/qdrant_storage")

# ── FastMCP app ───────────────────────────────────────────────────────────────
mcp = FastMCP("earnings-call-server")


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
        boost_recency: When True, rerank by blending semantic similarity with a
                       recency decay so newer calls surface higher.

    Returns:
        List of matching transcript chunks with metadata and relevance scores.
    """
    try:
        query_vector = embed_query(query)

        # Build filter conditions.
        # `date` is DATETIME-indexed → use DatetimeRange, not numeric Range.
        conditions: list[Condition] = []

        if ticker:
            conditions.append(
                FieldCondition(key="ticker", match=MatchValue(value=ticker.upper()))
            )

        if date_range:
            parts = date_range.split(":")
            if len(parts) == 2:
                start_dt = datetime.fromisoformat(parts[0].strip()).replace(tzinfo=timezone.utc)
                end_dt = datetime.fromisoformat(parts[1].strip()).replace(tzinfo=timezone.utc)
                conditions.append(
                    FieldCondition(
                        key="date",
                        range=DatetimeRange(gte=start_dt, lte=end_dt),
                    )
                )

        qdrant_filter = Filter(must=conditions) if conditions else None

        if boost_recency:
            # Prefetch a wider candidate pool by pure similarity, then
            # rerank with: final = $score + WEIGHT * exp_decay(now − date)
            # The DATETIME `date` field powers the decay expression.
            now_iso = datetime.now(timezone.utc).isoformat()
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

        return [
            {
                "point_id": str(r.id),
                "ticker": payload.get("ticker"),
                "company": payload.get("company"),
                "quarter": payload.get("quarter"),
                "year": payload.get("year"),
                "chunk_text": payload.get("chunk_text"),
                "speaker": payload.get("speaker"),
                "start_time": payload.get("start_time"),
                "score": r.score,
            }
            for r in results.points
            for payload in [r.payload or {}]
        ]

    except Exception as exc:
        return [{"error": str(exc)}]


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: get_audio_clip
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_audio_clip(point_id: str) -> dict[str, Any]:
    """
    Retrieve a base64-encoded audio clip for a specific transcript chunk.

    Tries pre-sliced clips first; falls back to on-the-fly slicing via pydub
    if the full audio file is present and pydub is installed.

    Args:
        point_id: UUID of the Qdrant point returned by search_earnings.

    Returns:
        Dict with audio_base64 and metadata, or {"error": "..."} on failure.
    """
    try:
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        if not points:
            return {"error": f"Point {point_id} not found in collection"}

        payload = points[0].payload or {}
        ticker: str = payload.get("ticker", "")
        start_time: float = payload.get("start_time", 0.0)
        end_time: float = payload.get("end_time", 0.0)

        clip_path = CLIPS_DIR / f"{point_id}.mp3"

        if clip_path.exists():
            audio_b64 = base64.b64encode(clip_path.read_bytes()).decode()
            return {
                "point_id": point_id,
                "ticker": ticker,
                "audio_base64": audio_b64,
                "start_time": start_time,
                "end_time": end_time,
                "format": "mp3",
            }

        # On-the-fly slicing from the full earnings call audio
        audio_file = payload.get("audio_file", "")
        full_audio_path = AUDIO_DIR / audio_file

        if full_audio_path.exists():
            try:
                from pydub import AudioSegment  # type: ignore

                audio = AudioSegment.from_mp3(str(full_audio_path))
                clip = audio[int(start_time * 1000):int(end_time * 1000)]
                CLIPS_DIR.mkdir(parents=True, exist_ok=True)
                clip.export(str(clip_path), format="mp3")
                audio_b64 = base64.b64encode(clip_path.read_bytes()).decode()
                return {
                    "point_id": point_id,
                    "ticker": ticker,
                    "audio_base64": audio_b64,
                    "start_time": start_time,
                    "end_time": end_time,
                    "format": "mp3",
                }
            except ImportError:
                pass

        return {
            "error": (
                f"No pre-sliced clip found for point {point_id} "
                f"(expected: {clip_path}). "
                "Run the ingestion pipeline to generate clips, or install "
                "pydub for on-the-fly slicing."
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
    Return news articles and web context relevant to an earnings call chunk.

    Checks a local cache first; if not cached and ASKNEWS_API_KEY is set,
    issues a live AskNews deep-research call and caches the result.

    Args:
        point_id: UUID of the Qdrant point returned by search_earnings.

    Returns:
        Dict with ticker, date, analysis, and articles list.
    """
    try:
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        if not points:
            return {"error": f"Point {point_id} not found"}

        payload = points[0].payload or {}
        ticker: str = payload.get("ticker", "")
        company: str = payload.get("company", "")
        quarter: str = payload.get("quarter", "")
        year: int = payload.get("year", 0)
        date: str = payload.get("date", "")
        speaker: str = payload.get("speaker", "")
        chunk_text: str = payload.get("chunk_text", "")

        cache_path = ASKNEWS_CACHE_DIR / f"{ticker}_{date}_{point_id}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        asknews_key = os.getenv("ASKNEWS_API_KEY", "")
        if not asknews_key:
            return {
                "ticker": ticker,
                "date": date,
                "articles": [],
                "note": (
                    "No cached news found and ASKNEWS_API_KEY not set. "
                    "Run ingest/04_build_asknews_context.py to pre-populate the cache."
                ),
            }

        from asknews_sdk import AskNewsSDK  # type: ignore
        from asknews_sdk.dto.deepnews import (  # type: ignore
            AnthropicTextDelta,
            ContentBlockDeltaEvent,
            CreateDeepNewsResponseStreamChunkV2,
            CreateDeepNewsResponseStreamSource,
            CreateDeepNewsResponseStreamSourcesNewsSource,
            CreateDeepNewsResponseStreamSourcesWebSource,
        )

        ask = AskNewsSDK(api_key=asknews_key)
        call_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        query = (
            f"Use the search_news, search_x_twitter, search_wikipedia, and search_google "
            f"tools to find information relevant to this moment from the "
            f"{company} ({ticker}) {quarter} {year} earnings call on {date}.\n\n"
            f"The speaker is {speaker}, and they said:\n\"{chunk_text}\"\n\n"
            f"Search for news/tweets ±7 days around {call_dt.date()} that explains "
            f"the macro events, market conditions, or company-specific news that "
            f"provides context for what {speaker} was discussing."
        )

        response = ask.chat.get_deep_news(
            messages=[{"role": "user", "content": query}],
            search_depth=1,
            max_depth=4,
            sources=["asknews", "google", "x", "wiki"],
            stream=True,
            return_sources=True,
            model="claude-sonnet-4-6",
            engine="v2.0",
            only_cited_sources=True,
        )

        entity_types = {
            "Person", "Organization", "Location", "Event", "Money",
            "Law", "Politics", "Product", "Technology", "Science",
        }

        full_text_parts: list[str] = []
        articles: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for message in response:
            if isinstance(message, CreateDeepNewsResponseStreamChunkV2):
                event = message.choices[0].delta
                if isinstance(event, ContentBlockDeltaEvent) and isinstance(event.delta, AnthropicTextDelta):
                    full_text_parts.append(event.delta.text)
                continue

            if not isinstance(message, CreateDeepNewsResponseStreamSource):
                continue

            if isinstance(message.source, CreateDeepNewsResponseStreamSourcesNewsSource):
                item = message.source.data
                article_id = str(item.article_id)
                if article_id in seen_ids:
                    continue
                entities = {
                    k: v for k, v in item.entities.model_dump().items()
                    if k in entity_types and v
                }
                articles.append({
                    "title": item.eng_title or item.title,
                    "summary": item.summary,
                    "sentiment": item.sentiment,
                    "entities": entities,
                    "source": item.source_id,
                    "url": str(item.article_url),
                    "published_at": str(item.pub_date),
                    "content_type": item.content_type,
                })
                seen_ids.add(article_id)

            elif isinstance(message.source, CreateDeepNewsResponseStreamSourcesWebSource):
                item = message.source.data
                article_id = str(item.url)
                if article_id in seen_ids:
                    continue
                articles.append({
                    "title": item.title,
                    "summary": " ".join(item.key_points) if item.key_points else item.raw_text,
                    "source": item.source,
                    "url": str(item.url),
                    "published_at": item.published,
                    "content_type": "web",
                })
                seen_ids.add(article_id)

        full_text = "".join(full_text_parts)
        tag_open, tag_close = "<final_answer>", "</final_answer>"
        start = full_text.find(tag_open)
        end = full_text.find(tag_close)
        analysis = (
            full_text[start + len(tag_open):end].strip()
            if start != -1 and end != -1
            else full_text.strip()
        )

        result = {
            "ticker": ticker,
            "date": date,
            "window": f"{(call_dt - timedelta(days=7)).date()} → {call_dt.date()}",
            "analysis": analysis,
            "articles": articles,
        }

        ASKNEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
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
    Find transcript chunks semantically similar to a given chunk.

    Retrieves the seed point's stored text vector and searches for nearest
    neighbours, excluding the seed itself.

    Args:
        point_id: UUID of the Qdrant point to use as the reference.

    Returns:
        List of up to 5 similar chunks with point_id, ticker, chunk_text,
        score, quarter, year.
    """
    try:
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

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=seed_text,
            using="text",
            query_filter=Filter(must_not=[HasIdCondition(has_id=[point_id])]),
            limit=5,
            with_payload=True,
        )

        return [
            {
                "point_id": str(r.id),
                "ticker": payload.get("ticker"),
                "chunk_text": payload.get("chunk_text"),
                "score": r.score,
                "quarter": payload.get("quarter"),
                "year": payload.get("year"),
            }
            for r in results.points
            for payload in [r.payload or {}]
        ]

    except Exception as exc:
        return [{"error": str(exc)}]


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5: compare_tickers  (new contribution)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def compare_tickers(
    query: str,
    tickers: list[str],
    limit_per_ticker: int = 3,
    boost_recency: bool = False,
) -> dict[str, Any]:
    """
    Compare what multiple companies said about the same topic.

    Runs search_earnings for each ticker in parallel (sequential calls, shared
    embedding) and returns a side-by-side dict so Claude can synthesise a
    cross-company view without making N separate tool calls.

    Args:
        query:            Topic to compare, e.g. "AI infrastructure investment"
        tickers:          Companies to compare, e.g. ["NVDA", "AAPL", "AMZN"]
        limit_per_ticker: Results per company (default 3, max 10)
        boost_recency:    Apply recency decay reranking for each ticker search

    Returns:
        {
          "query": "...",
          "results": {
            "NVDA": [{"chunk_text": ..., "score": ..., ...}, ...],
            "AAPL": [...],
          }
        }
    """
    try:
        # Embed once, reuse across all ticker searches
        query_vector = embed_query(query)
        limit = max(1, min(limit_per_ticker, 10))
        now_iso = datetime.now(timezone.utc).isoformat()

        per_ticker: dict[str, list[dict[str, Any]]] = {}

        for ticker in tickers:
            ticker_upper = ticker.upper()
            conditions: list[Condition] = [
                FieldCondition(key="ticker", match=MatchValue(value=ticker_upper))
            ]
            qdrant_filter = Filter(must=conditions)

            if boost_recency:
                results = client.query_points(
                    collection_name=COLLECTION_NAME,
                    prefetch=Prefetch(
                        query=query_vector,
                        using="text",
                        filter=qdrant_filter,
                        limit=limit * 6,
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
                    limit=limit,
                    with_payload=True,
                )
            else:
                results = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    using="text",
                    query_filter=qdrant_filter,
                    limit=limit,
                    with_payload=True,
                )

            per_ticker[ticker_upper] = [
                {
                    "point_id": str(r.id),
                    "company": payload.get("company"),
                    "quarter": payload.get("quarter"),
                    "year": payload.get("year"),
                    "date": payload.get("date"),
                    "chunk_text": payload.get("chunk_text"),
                    "speaker": payload.get("speaker"),
                    "score": r.score,
                }
                for r in results.points
                for payload in [r.payload or {}]
            ]

        return {"query": query, "results": per_ticker}

    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Bonus Tool 6: get_sec_filings
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_sec_filings(ticker: str, year: int) -> dict[str, Any]:
    """
    Scrape recent SEC filings (10-Q, 10-K) for a given ticker via Playwright.

    Args:
        ticker: Stock ticker, e.g. "NVDA"
        year:   Calendar year, e.g. 2024

    Returns:
        Dict with ticker and a list of filing dicts (type, date, url).
    """
    try:
        from browser_agent.sec_scraper import scrape_sec_filings as _scrape  # type: ignore

        return _scrape(ticker=ticker, year=year)
    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
