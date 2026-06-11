# WORKSHOP EXERCISE: Complete the TODO sections below
"""
Earnings Call MCP Server — skeleton for workshop participants.

Your job: implement the four MCP tools so that Claude can query the Qdrant
collection of earnings call transcripts.

How to run once you're done:
    python mcp_server/server.py

To register with Claude Desktop / Claude Code first run:
    python cli/setup_mcp.py install
"""

import base64
import json
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import models

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from the workshop repo even when an MCP client
# launches this server from a different working directory.
load_dotenv(REPO_ROOT / ".env")

# ── Third-party imports ──────────────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient

# Local embedding helper (handles caching for offline mode)
from mcp_server.embeddings import embed_query
from research.earnings_research import run_research_analysis

# ── Configuration (from .env) ─────────────────────────────────────────────────
QDRANT_URL: str | None = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH: str | None = os.getenv("QDRANT_PATH") or None
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "earnings_calls")
CLIPS_DIR: Path = Path(os.getenv("CLIPS_DIR", str(REPO_ROOT / "data" / "audio_clips")))
ASKNEWS_CACHE_DIR: Path = REPO_ROOT / "data" / "asknews_cache"
AUDIO_DIR: Path = Path(os.getenv("AUDIO_DIR", str(REPO_ROOT / "data" / "audio")))
RECENCY_BOOST_WEIGHT = 0.3
RECENCY_HALF_LIFE_DAYS = 180

# ── Qdrant client ────────────────────────────────────────────────────────────
if QDRANT_URL:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        check_compatibility=False,
    )
else:
    client = QdrantClient(path=QDRANT_PATH or str(REPO_ROOT / "data" / "qdrant_storage"))

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
                       e.g. "2023-01-01:2024-01-01"
        boost_recency: When true, rerank semantic candidates with a date-decay
                       bonus so newer calls surface slightly higher.

    Returns:
        List of matching transcript chunks with metadata and relevance scores.
    """
    try:
        query_vector = embed_query(query)

        filters: list[models.Condition] = []
        if ticker:
            filters.append(
                models.FieldCondition(
                    key="ticker",
                    match=models.MatchValue(value=ticker.upper()),
                )
            )

        if date_range:
            parts = [part.strip() for part in date_range.split(":", 1)]
            if len(parts) != 2 or not all(parts):
                raise ValueError('date_range must be "YYYY-MM-DD:YYYY-MM-DD"')
            start_date, end_date = parts
            filters.append(
                models.FieldCondition(
                    key="date",
                    range=models.DatetimeRange(gte=start_date, lte=end_date),
                )
            )

        qdrant_filter = models.Filter(must=filters) if filters else None

        if boost_recency:
            now_iso = datetime.now(timezone.utc).isoformat()
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=models.Prefetch(
                    query=query_vector,
                    using="text",
                    filter=qdrant_filter,
                    limit=30,
                ),
                query=models.FormulaQuery(
                    formula=models.SumExpression(
                        sum=[
                            "$score",
                            models.MultExpression(
                                mult=[
                                    RECENCY_BOOST_WEIGHT,
                                    models.ExpDecayExpression(
                                        exp_decay=models.DecayParamsExpression(
                                            x=models.DatetimeKeyExpression(
                                                datetime_key="date"
                                            ),
                                            target=models.DatetimeExpression(
                                                datetime=now_iso
                                            ),
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
                "end_time": payload.get("end_time"),
                "date": payload.get("date"),
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

    Looks up the point in Qdrant to get its time offsets, then returns the
    pre-sliced audio clip from data/audio_clips/{point_id}.mp3 if it exists.

    Args:
        point_id: UUID of the Qdrant point returned by search_earnings.

    Returns:
        Dict with keys: point_id, ticker, audio_base64, start_time,
        end_time, format.  On error, returns {"error": "<message>"}.
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
        start_time = float(payload.get("start_time") or 0.0)
        end_time = float(payload.get("end_time") or 0.0)
        clip_path = CLIPS_DIR / f"{point_id}.mp3"

        if not clip_path.exists():
            audio_file = payload.get("audio_file")
            full_audio_path = AUDIO_DIR / audio_file if audio_file else None
            if full_audio_path and full_audio_path.exists():
                from pydub import AudioSegment  # type: ignore

                CLIPS_DIR.mkdir(parents=True, exist_ok=True)
                audio = AudioSegment.from_mp3(str(full_audio_path))
                audio[int(start_time * 1000) : int(end_time * 1000)].export(
                    str(clip_path),
                    format="mp3",
                )

        if not clip_path.exists():
            return {
                "error": (
                    f"No audio clip found for {point_id}. Expected {clip_path} "
                    "or a full source MP3 in data/audio."
                )
            }

        return {
            "point_id": point_id,
            "ticker": payload.get("ticker"),
            "audio_base64": base64.b64encode(clip_path.read_bytes()).decode(),
            "start_time": start_time,
            "end_time": end_time,
            "format": "mp3",
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
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        if not points:
            return {"error": f"Point {point_id} not found in collection"}

        payload = points[0].payload or {}
        ticker = str(payload.get("ticker") or "")
        date = str(payload.get("date") or "")[:10]
        company = str(payload.get("company") or "")
        quarter = str(payload.get("quarter") or "")
        year = payload.get("year") or ""
        speaker = str(payload.get("speaker") or "")
        chunk_text = str(payload.get("chunk_text") or "")

        cache_paths = [
            ASKNEWS_CACHE_DIR / f"{ticker}_{date}_{point_id}.json",
            ASKNEWS_CACHE_DIR / f"{ticker}_{date}.json",
        ]
        for cache_path in cache_paths:
            if cache_path.exists():
                return json.loads(cache_path.read_text())

        asknews_key = os.getenv("ASKNEWS_API_KEY", "")
        if not asknews_key:
            return {
                "ticker": ticker,
                "date": date,
                "articles": [],
                "note": (
                    "No AskNews cache found and ASKNEWS_API_KEY is not set. "
                    "Run ingest/04_build_asknews_context.py or add a key for live context."
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
            f"Find context for this specific moment from the {company} ({ticker}) "
            f"{quarter} {year} earnings call on {date}.\n\n"
            f"Speaker: {speaker}\n"
            f"Quote: \"{chunk_text}\"\n\n"
            f"Search news, web, Wikipedia, and X/Twitter for company-specific or "
            f"macro context around {(call_dt - timedelta(days=7)).date()} through "
            f"{call_dt.date()}."
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

        articles: list[dict[str, Any]] = []
        seen_article_ids: set[str] = set()
        analysis_parts: list[str] = []
        entity_types = {
            "Person",
            "Organization",
            "Location",
            "Event",
            "Money",
            "Law",
            "Politics",
            "Product",
            "Technology",
            "Science",
        }

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
                entities = {
                    k: v
                    for k, v in item.entities.model_dump().items()
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

        analysis = "".join(analysis_parts).strip()
        result = {
            "ticker": ticker,
            "date": date,
            "window": f"{(call_dt - timedelta(days=7)).date()} -> {call_dt.date()}",
            "analysis": analysis,
            "articles": articles,
        }

        ASKNEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_paths[0].write_text(json.dumps(result, indent=2))
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

    Uses Qdrant's recommendation API (positive example = the given point)
    to surface related content, which may be from other quarters or tickers.

    Args:
        point_id: UUID of the Qdrant point to use as the reference.

    Returns:
        List of up to 5 similar chunks with point_id, ticker, chunk_text,
        score, quarter, year.  On error, returns [{"error": "<message>"}].
    """
    try:
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_vectors=True,
        )
        if not points:
            return [{"error": f"Point {point_id} not found in collection"}]

        vectors = points[0].vector
        if isinstance(vectors, dict):
            seed_vector = vectors.get("text")
        else:
            seed_vector = vectors

        if seed_vector is None:
            return [{"error": f"Point {point_id} has no text vector"}]

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=seed_vector,
            using="text",
            query_filter=models.Filter(
                must_not=[models.HasIdCondition(has_id=[point_id])]
            ),
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
# Research brief tool
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool()
def research_earnings(
    task_description: str,
    max_evidence: int = 12,
    use_semantic_search: bool = True,
    use_asknews: bool = True,
) -> dict[str, Any]:
    """
    Turn an open-ended research task into a finished earnings-call analysis.

    Use this for journalism/data-science prompts such as "research cyclic AI
    investment and find evidence in earnings calls." The tool builds a query
    plan, searches or scans the transcript corpus, ranks supporting chunks, writes
    an evidence package, and returns a finished Markdown analysis plus structured
    evidence with point IDs and audio paths.

    Args:
        task_description: Free-text research or analysis task.
        max_evidence: Maximum transcript chunks to include, clamped to 1..25.
        use_semantic_search: When true, use Qdrant semantic search plus local
            lexical scoring. When false, use local transcripts only.
        use_asknews: When true, augment the transcript analysis with a live
            AskNews DeepNews research pass if ASKNEWS_API_KEY is set.

    Returns:
        Dict with analysis_markdown, evidence_package, the structured evidence
        brief, and optional corpus theme evidence.
    """
    try:
        bounded_max = max(1, min(int(max_evidence), 25))
        return run_research_analysis(
            task_description,
            max_evidence=bounded_max,
            use_qdrant=use_semantic_search,
            use_asknews=use_asknews,
        )
    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
