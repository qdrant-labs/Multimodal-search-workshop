# INSTRUCTOR REFERENCE: Full working solution
"""
Earnings Call MCP Server — complete implementation.

This file is the reference solution for the workshop.  Participants work in
server.py; this file demonstrates what a finished implementation looks like.

Run with:
    python mcp_server/server_solution.py
"""

import base64
import json
import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from datetime import datetime, timezone

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

# Ensure ffmpeg/ffprobe are available for audio slicing
if not shutil.which("ffmpeg"):
    try:
        import static_ffmpeg  # type: ignore
        static_ffmpeg.add_paths()
    except ImportError:
        pass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcp_server.embeddings import embed_query

# Recency boost: weight applied to the decay term, and the half-life
# (in days) at which an older call's recency bonus drops to half.
RECENCY_BOOST_WEIGHT = 0.3
RECENCY_HALF_LIFE_DAYS = 180

# ── Configuration ─────────────────────────────────────────────────────────────
QDRANT_URL: str | None = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH: str | None = os.getenv("QDRANT_PATH") or None
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "earnings_calls")
CLIPS_DIR: Path = Path(os.getenv("CLIPS_DIR", "./data/audio_clips"))
ASKNEWS_CACHE_DIR: Path = Path("./data/asknews_cache")
AUDIO_DIR: Path = Path(os.getenv("AUDIO_DIR", "./data/audio"))

# ── Qdrant client ─────────────────────────────────────────────────────────────
if QDRANT_URL:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    client = QdrantClient(path=QDRANT_PATH or "./data/qdrant_storage")

# ── FastMCP app ────────────────────────────────────────────────────────────────
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
        boost_recency: When True, rerank by combining semantic similarity with a
                       time-decay bonus so more recent calls surface higher.

    Returns:
        List of matching transcript chunks with metadata and relevance scores.
    """
    try:
        # Step 1: Embed the query
        query_vector = embed_query(query)

        # Step 2: Build Qdrant filter.
        # `date` is a DATETIME-indexed field, so use DatetimeRange (NOT the
        # numeric Range) — it accepts ISO date strings like "2025-01-01".
        conditions: list[Condition] = []

        if ticker:
            conditions.append(
                FieldCondition(key="ticker", match=MatchValue(value=ticker.upper()))
            )

        if date_range:
            parts = date_range.split(":")
            if len(parts) == 2:
                start_date, end_date = (
                    datetime.fromisoformat(parts[0].strip()).replace(tzinfo=timezone.utc), 
                    datetime.fromisoformat(parts[1].strip()).replace(tzinfo=timezone.utc)
                )
                conditions.append(
                    FieldCondition(
                        key="date",
                        range=DatetimeRange(gte=start_date, lte=end_date),
                    )
                )

        qdrant_filter = Filter(must=conditions) if conditions else None

        # Step 3: Run the vector search against the `text` named vector.
        # The collection stores two named vectors per chunk (text + audio)
        # in the same multimodal space, so we must pick which one to query.
        if boost_recency:
            # Recency boosting via the Query API formula:
            #   final = $score + WEIGHT * exp_decay(now - call_date)
            # Prefetch pulls a wider candidate pool by pure similarity, then
            # the formula reranks them. The decay reads the DATETIME `date`
            # field; midpoint=0.5 means the bonus halves at one half-life.
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

        # Step 4: Format results
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

    Args:
        point_id: UUID of the Qdrant point returned by search_earnings.

    Returns:
        Dict with audio_base64 and metadata, or {"error": "..."} on failure.
    """
    try:
        # Step 1: Retrieve point from Qdrant
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        if not points:
            return {"error": f"Point {point_id} not found in collection"}

        payload = points[0].payload
        if payload is None: 
            raise ValueError("Payload is not in the point return")
        
        ticker: str = payload.get("ticker", "")
        start_time: float = payload.get("start_time", 0.0)
        end_time: float = payload.get("end_time", 0.0)

        # Step 2: Check for a pre-sliced clip
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

        # Step 3: Try to slice from the full audio file using pydub
        audio_file = payload.get("audio_file", "")
        full_audio_path = AUDIO_DIR / audio_file

        if full_audio_path.exists():
            try:
                from pydub import AudioSegment  # type: ignore

                audio = AudioSegment.from_mp3(str(full_audio_path))
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                clip = audio[start_ms:end_ms]

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
                pass  # pydub not available; fall through to error

        return {
            "error": (
                f"No pre-sliced clip found for point {point_id} "
                f"(expected: {clip_path}). "
                "Run the ingestion pipeline to generate clips, or ensure "
                "pydub is installed for on-the-fly slicing."
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
    Return news articles, tweets, and web context relevant to the specific 
    earnings call chunk identified by point_id.

    Args:
        point_id: UUID of the Qdrant point returned by search_earnings.

    Returns:
        Dict with ticker, date, window, and articles list, deepnews analysis
    """
    try:
        from datetime import datetime, timedelta, timezone  # noqa: PLC0415

        # Step 1: Retrieve point
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        if not points:
            return {"error": f"Point {point_id} not found"}

        payload = points[0].payload
        if payload is None: 
            raise ValueError("Payload is not in the point return")
        ticker: str = payload.get("ticker", "")
        company: str = payload.get("company", "")
        quarter: str = payload.get("quarter", "")
        year: int = payload.get("year", 0)
        date: str = payload.get("date", "")      # "YYYY-MM-DD"
        speaker: str = payload.get("speaker", "")
        chunk_text: str = payload.get("chunk_text", "")

        # Step 2: Check disk cache
        cache_path = ASKNEWS_CACHE_DIR / f"{ticker}_{date}_{point_id}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        # Step 3: Live AskNews deep research call
        asknews_key = os.getenv("ASKNEWS_API_KEY", "")
        if not asknews_key:
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
            f"Use the search_news, search_x_twitter, search_wikipedia, and search_google tools to "
            f"search for information relevant to this specific moment from the "
            f"{company} ({ticker}) {quarter} {year} earnings call on {date}.\n\n"
            f"The speaker is {speaker}, and they said:\n\"{chunk_text}\"\n\n"
            f"Search for news/tweets ±7 days around {call_dt.date()} that explains the macro events, "
            f"market conditions, or company-specific news that provides context for what "
            f"{speaker} was discussing. Also search for any relevant background in the news, "
            f"google, wikipedia, and twitter from the prior couple of months."
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
        seen_article_ids: set[str] = set()
        for message in response:
            if isinstance(message, CreateDeepNewsResponseStreamChunkV2):
                event = message.choices[0].delta
                if isinstance(event, ContentBlockDeltaEvent) and isinstance(event.delta, AnthropicTextDelta):
                    full_text_parts.append(event.delta.text)
                continue

            if not isinstance(message, CreateDeepNewsResponseStreamSource):
                continue

            if isinstance(message.source, CreateDeepNewsResponseStreamSourcesNewsSource):
                # data is SearchResponseDictItem (extends Article)
                item = message.source.data
                article_id = str(item.article_id)
                if article_id in seen_article_ids:
                    continue
                entities = {
                    k: v for k, v in item.entities.model_dump().items()
                    if k in entity_types and v
                }
                articles.append(
                    {
                        "title": item.eng_title or item.title,
                        "summary": item.summary,
                        "sentiment": item.sentiment,
                        "entities": entities,
                        "language": item.language,
                        "bias": item.bias,
                        "reporting_voice": item.reporting_voice,
                        "source": item.source_id,
                        "authors": [a.model_dump() for a in (item.authors or [])],
                        "content_type": item.content_type,
                        "url": str(item.article_url),
                        "image_url": str(item.image_url or ""),
                        "image_description": item.image_description or "",
                        "published_at": str(item.pub_date),
                    }
                )
                seen_article_ids.add(article_id)

            elif isinstance(message.source, CreateDeepNewsResponseStreamSourcesWebSource):
                # data is WebSearchResult: title, url, source, published, key_points, raw_text
                item = message.source.data
                article_id = str(item.url)
                if article_id in seen_article_ids:
                    continue
                articles.append(
                    {
                        "title": item.title,
                        "summary": " ".join(item.key_points) if item.key_points else item.raw_text,
                        "sentiment": None,
                        "entities": {},
                        "language": "",
                        "bias": None,
                        "reporting_voice": "",
                        "source": item.source,
                        "authors": [],
                        "content_type": "web",
                        "url": str(item.url),
                        "image_url": "",
                        "image_description": "",
                        "published_at": item.published,
                    }
                )
                seen_article_ids.add(article_id)

        full_text = "".join(full_text_parts)
        tag_open = "<final_answer>"
        tag_close = "</final_answer>"
        start = full_text.find(tag_open)
        end = full_text.find(tag_close)
        analysis = full_text[start + len(tag_open):end].strip() if start != -1 and end != -1 else full_text.strip()

        result = {
            "ticker": ticker,
            "date": date,
            "window": f"{(call_dt - timedelta(days=7)).date()} → {call_dt.date()}",
            "analysis": analysis,
            "articles": articles,
        }

        # Cache for offline use
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

    Uses Qdrant's recommendation API with the given point as a positive example.

    Args:
        point_id: UUID of the Qdrant point to use as the reference.

    Returns:
        List of up to 5 similar chunks.
    """
    try:
        # Fetch the seed point's vectors. For named-vector collections
        # `retrieve(..., with_vectors=True)` returns pts[0].vector as a dict
        # like {"text": [...], "audio": [...]}. Use the text vector for
        # cross-call topical similarity.
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
# Bonus Tool 5: scrape_sec_filings (browser agent demo)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def scrape_sec_filings(ticker: str, year: int) -> dict[str, Any]:
    """
    Scrape recent SEC filings (10-Q, 10-K) for a given ticker using Playwright.

    This tool demonstrates how a browser agent can be wired into an MCP server.

    Args:
        ticker: Stock ticker, e.g. "NVDA"
        year:   Calendar year to search, e.g. 2024

    Returns:
        Dict with ticker and a list of filing dicts (type, date, url).
    """
    try:
        from browser_agent.sec_scraper import scrape_sec_filings as _scrape

        return _scrape(ticker=ticker, year=year)
    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
