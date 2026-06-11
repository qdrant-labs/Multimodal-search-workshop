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
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import models

# Load environment variables from .env
load_dotenv()

# ── Third-party imports ──────────────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient

# Local embedding helper (handles caching for offline mode)
from mcp_server.embeddings import embed_query

# ── Configuration (from .env) ─────────────────────────────────────────────────
QDRANT_URL: str | None = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH: str | None = os.getenv("QDRANT_PATH") or None
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "earnings_calls")
CLIPS_DIR: Path = Path(os.getenv("CLIPS_DIR", "./data/audio_clips"))
ASKNEWS_CACHE_DIR: Path = Path("./data/asknews_cache")

# Recency boost (Exercise 6): weight applied to the time-decay term, and the
# half-life in days at which an older call's recency bonus drops to half.
RECENCY_BOOST_WEIGHT = 0.3
RECENCY_HALF_LIFE_DAYS = 180

# ── Qdrant client ────────────────────────────────────────────────────────────
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
        query:      Natural-language question, e.g. "data center demand outlook"
        ticker:     Optional stock ticker to restrict results, e.g. "NVDA"
        date_range: Optional ISO date range "YYYY-MM-DD:YYYY-MM-DD"
                    e.g. "2023-01-01:2024-01-01"
        boost_recency: When True, rerank results by blending similarity with an
                    exponential time-decay bonus so more recent calls rank higher.

    Returns:
        List of matching transcript chunks with metadata and relevance scores.
    """
    # TODO Step 1: Embed the query text into a vector.
    #   Use the embed_query() function imported above.
    #   It returns a list[float] of length 3072 from Gemini Embedding 2
    #   (multimodal — text and audio share this vector space).
    #
    query_vector = embed_query(query)

    # TODO Step 2: Build an optional Qdrant filter.
    #   You only need a filter when ticker or date_range is provided.
    #   Qdrant filter structure:
    #
    #   from qdrant_client.models import Filter, FieldCondition, MatchValue, DatetimeRange
    #
    #   To filter by ticker:
    #       FieldCondition(key="ticker", match=MatchValue(value=ticker))
    #
    #   To filter by date range — the "date" payload field is indexed as
    #   DATETIME, so use DatetimeRange (NOT the numeric Range). It accepts
    #   ISO date strings:
    #       Parse date_range.split(":") → [start_date, end_date]
    #       FieldCondition(key="date", range=DatetimeRange(gte=start_date, lte=end_date))
    #
    #   Wrap conditions in Filter(must=[...]) if you have any.
    #
    #   STRETCH — recency boosting: add a `boost_recency: bool = False` arg and,
    #   when set, rerank with the Query API formula
    #       final = $score + 0.3 * exp_decay(now - date)
    #   using prefetch + FormulaQuery. See implementation_guide.md (Exercise 5)
    #   and server_solution.py for the full pattern.
    filters: list[models.Condition] = []
    if ticker:
        filters.append(
            models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker))
        )

    if date_range:
        start_date, end_date = [
            datetime.fromisoformat(date) for date in date_range.split(":")
        ]

        filters.append(
            models.FieldCondition(
                key="date", range=models.DatetimeRange(gte=start_date, lte=end_date)
            )
        )

    # TODO Step 3: Run the vector search.
    #   The collection stores TWO named vectors per chunk — `text` and
    #   `audio` — both produced by gemini-embedding-2 in the same shared
    #   space. Pass `using="text"` so Qdrant searches against the text
    #   vector. Swap to `using="audio"` for audio→audio retrieval later.
    #
    #   results = client.query_points(
    #       collection_name=COLLECTION_NAME,
    #       query=query_vector,
    #       using="text",
    #       query_filter=qdrant_filter,   # None if no filter
    #       limit=5,
    #       with_payload=True,
    #   )
    #
    #   Each result has: .id, .score, .payload (dict)

    qdrant_filter = models.Filter(must=filters) if filters else None

    if boost_recency:
        # Recency boost via the Query API formula:
        #   final = $score + RECENCY_BOOST_WEIGHT * exp_decay(now - call_date)
        # Prefetch pulls a wider candidate pool by pure similarity (limit 30 >
        # final limit 5), then the formula reranks it. The decay reads the
        # DATETIME `date` field; midpoint=0.5 halves the bonus at one half-life.
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
                                        x=models.DatetimeKeyExpression(datetime_key="date"),
                                        target=models.DatetimeExpression(datetime=now_iso),
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
            "ticker": r.payload.get("ticker"),
            "company": r.payload.get("company"),
            "quarter": r.payload.get("quarter"),
            "year": r.payload.get("year"),
            "chunk_text": r.payload.get("chunk_text"),
            "speaker": r.payload.get("speaker"),
            "start_time": r.payload.get("start_time"),
            "score": r.score,
        }
        for r in results.points
        if r.payload
    ]

    # TODO Step 4: Format the results.
    #   Return a list of dicts, one per result, containing:
    #       point_id, ticker, company, quarter, year,
    #       chunk_text, speaker, start_time, score
    #
    #   Example:
    #   return [
    #       {
    #           "point_id": str(r.id),
    #           "ticker": r.payload.get("ticker"),
    #           "company": r.payload.get("company"),
    #           "quarter": r.payload.get("quarter"),
    #           "year": r.payload.get("year"),
    #           "chunk_text": r.payload.get("chunk_text"),
    #           "speaker": r.payload.get("speaker"),
    #           "start_time": r.payload.get("start_time"),
    #           "score": r.score,
    #       }
    #       for r in results.points
    #   ]

    # Remove this placeholder once you've implemented the steps above
    # return [{"error": "search_earnings is not yet implemented — complete the TODOs!"}]


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
    # TODO: Implement get_audio_clip
    #
    # Step 1: Retrieve the point from Qdrant
    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_payload=True,
    )
    if not points:
        return {"error": f"Point {point_id} not found"}
    payload = points[0].payload
    #
    # Step 2: Check if a pre-sliced clip exists
    clip_path = CLIPS_DIR / f"{point_id}.mp3"
    audio_b64 = ""
    if clip_path.exists():
        audio_b64 = base64.b64encode(clip_path.read_bytes()).decode()
        return {
            "point_id": point_id,
            "ticker": payload.get("ticker") if payload else None,
            "audio_base64": audio_b64,
            "start_time": payload.get("start_time") if payload else None,
            "end_time": payload.get("end_time") if payload else None,
            "format": "mp3",
        }
    #
    # Step 3: If no clip file, return a helpful error
    return {
        "error": (
            f"No pre-sliced clip found for {point_id}. "
            "Run the ingestion pipeline or slice manually."
        )
    }

    return {"error": "get_audio_clip is not yet implemented — complete the TODOs!"}


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
        # Step 1: Retrieve the point — same pattern as get_audio_clip.
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        if not points:
            return {"error": f"Point {point_id} not found"}
        payload = points[0].payload or {}

        # Step 2: A chunk's news context is defined by WHO (ticker) and WHEN (date).
        # The live DeepNews query below also uses company/quarter/speaker/text.
        ticker = payload.get("ticker", "")
        date = payload.get("date", "")  # "YYYY-MM-DD"
        company = payload.get("company", "")
        quarter = payload.get("quarter", "")
        year = payload.get("year", 0)
        speaker = payload.get("speaker", "")
        chunk_text = payload.get("chunk_text", "")

        # Step 3: Cache-first. The ingest pipeline pre-fetches news per chunk as
        # {ticker}_{date}_{point_id}.json (older runs used {ticker}_{date}.json).
        cache_path = ASKNEWS_CACHE_DIR / f"{ticker}_{date}_{point_id}.json"
        if not cache_path.exists():
            alt = ASKNEWS_CACHE_DIR / f"{ticker}_{date}.json"
            if alt.exists():
                cache_path = alt
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        # Step 4: Cache miss. Only call the live API if a key is configured;
        # otherwise degrade gracefully — a tool should never hard-crash Claude.
        if not os.getenv("ASKNEWS_API_KEY"):
            return {
                "ticker": ticker,
                "date": date,
                "articles": [],
                "note": (
                    "No cached news for this chunk and ASKNEWS_API_KEY is not set. "
                    "Run ingest/04_build_asknews_context.py to pre-populate the cache, "
                    "or see server_solution.py for the live AskNews DeepNews call."
                ),
            }

        # Step 5 (live): stream AskNews DeepNews for this exact moment, collect the
        # cited news + web sources, extract the analysis, cache it, and return.
        from datetime import timedelta, timezone

        from asknews_sdk import AskNewsSDK  # type: ignore
        from asknews_sdk.dto.deepnews import (  # type: ignore
            AnthropicTextDelta,
            ContentBlockDeltaEvent,
            CreateDeepNewsResponseStreamChunkV2,
            CreateDeepNewsResponseStreamSource,
            CreateDeepNewsResponseStreamSourcesNewsSource,
            CreateDeepNewsResponseStreamSourcesWebSource,
        )

        ask = AskNewsSDK(api_key=os.getenv("ASKNEWS_API_KEY", ""))
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
                if isinstance(event, ContentBlockDeltaEvent) and isinstance(
                    event.delta, AnthropicTextDelta
                ):
                    full_text_parts.append(event.delta.text)
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
                        "summary": " ".join(item.key_points)
                        if item.key_points
                        else item.raw_text,
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
        analysis = (
            full_text[start + len(tag_open) : end].strip()
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

        # Cache to disk so this chunk's news is instant (and offline) next time.
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
        # Step 1: Fetch the seed point's STORED vectors — no re-embedding needed.
        # For named-vector collections, retrieve(..., with_vectors=True) returns
        # .vector as a dict {"text": [...], "audio": [...]}. Use the text vector
        # for topical ("more like this") similarity across calls and tickers.
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

        # Step 2: Search with that vector, excluding the seed itself — otherwise
        # the seed is always the #1 hit at score 1.0. must_not + HasIdCondition
        # filters it out.
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=seed_text,
            using="text",
            query_filter=models.Filter(
                must_not=[models.HasIdCondition(has_id=[point_id])]
            ),
            limit=5,
            with_payload=True,
        )

        # Step 3: Format (note results.points, not results).
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
# Tool 5: transcribe_audio  (voice prompt / uploaded conversation → text)
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool()
def transcribe_audio(audio_path: str, diarize: bool = True) -> dict[str, Any]:
    """
    Transcribe a spoken audio file into text with Gemini, so a voice prompt or
    an uploaded conversation can be searched or fact-checked against the
    earnings-call corpus.

    Typical fact-check flow: transcribe_audio(recording) → pull out the factual
    claims → search_earnings(claim) for each → compare claim vs. retrieved
    evidence → cite the real moment with get_audio_clip / get_news_context.

    Args:
        audio_path: Path to a local audio file (mp3/wav/m4a/…). Keep it short
                    (a few minutes); inline audio is capped near 18 MB.
        diarize:    When True, ask Gemini to label distinct speakers.

    Returns:
        Dict with keys: path, format, transcript. On error {"error": "..."}.
    """
    try:
        clip = Path(audio_path).expanduser()
        if not clip.exists():
            return {"error": f"Audio file not found: {audio_path}"}

        audio_bytes = clip.read_bytes()
        if len(audio_bytes) > 18_000_000:
            return {
                "error": (
                    f"Audio is {len(audio_bytes) // 1_000_000} MB; inline transcription "
                    "is capped near 18 MB. Use a shorter clip (longer recordings need "
                    "the Gemini Files API)."
                )
            }

        mime_map = {
            ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
            ".ogg": "audio/ogg", ".flac": "audio/flac", ".aac": "audio/aac",
        }
        mime = mime_map.get(clip.suffix.lower(), "audio/mpeg")

        from google import genai
        from google.genai import types

        gclient = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
        instruction = (
            "Transcribe this audio verbatim. "
            + (
                "Label each distinct speaker as 'Speaker 1:', 'Speaker 2:', etc. "
                if diarize
                else ""
            )
            + "Return only the transcript text, with no preamble or commentary."
        )
        resp = gclient.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                instruction,
            ],
        )
        return {
            "path": str(clip),
            "format": clip.suffix.lstrip(".").lower(),
            "transcript": (resp.text or "").strip(),
        }

    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
