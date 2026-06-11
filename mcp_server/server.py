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
from datetime import datetime
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
) -> list[dict[str, Any]]:
    """
    Semantic search over earnings call transcripts stored in Qdrant.

    Args:
        query:      Natural-language question, e.g. "data center demand outlook"
        ticker:     Optional stock ticker to restrict results, e.g. "NVDA"
        date_range: Optional ISO date range "YYYY-MM-DD:YYYY-MM-DD"
                    e.g. "2023-01-01:2024-01-01"

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
                key="date", range=models.DatetimeRange(lte=start_date, gte=end_date)
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

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=models.Filter(must=filters),
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
    # TODO: Implement get_news_context
    #
    # Step 1: Retrieve the point from Qdrant (same as in get_audio_clip)
    #
    # Step 2: Extract ticker and date from payload
    #   ticker = payload.get("ticker", "")
    #   date   = payload.get("date", "")
    #
    # Step 3: Build the expected cache file path
    #   cache_path = ASKNEWS_CACHE_DIR / f"{ticker}_{date}.json"
    #
    # Step 4a: If cache exists, load and return it
    #   if cache_path.exists():
    #       return json.loads(cache_path.read_text())
    #
    # Step 4b: If not cached, try a live AskNews call
    #   (requires ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET in .env)
    #   from asknews_sdk import AskNewsSDK
    #   sdk = AskNewsSDK(
    #       client_id=os.getenv("ASKNEWS_CLIENT_ID"),
    #       client_secret=os.getenv("ASKNEWS_CLIENT_SECRET"),
    #   )
    #   Then search, format results, save to cache, and return.
    #
    # Step 4c: If no cache and no credentials, return an informative error

    return {"error": "get_news_context is not yet implemented — complete the TODOs!"}


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
    # TODO: Implement recommend_similar
    #
    # qdrant-client ≥1.9 removed recommend() — use query_points() instead:
    #
    # Step 1: Fetch the seed point's stored vectors.
    #   For named-vector collections, with_vectors=True returns a dict:
    #       pts[0].vector == {"text": [...], "audio": [...]}
    #
    #   pts = client.retrieve(
    #       collection_name=COLLECTION_NAME,
    #       ids=[point_id],
    #       with_vectors=True,
    #   )
    #   if not pts:
    #       return [{"error": f"Point {point_id} not found"}]
    #   seed_text = pts[0].vector["text"]   # use text-side for topical sim
    #
    # Step 2: Search for nearest neighbours, excluding the seed point itself
    #   from qdrant_client.models import Filter, HasIdCondition
    #
    #   results = client.query_points(
    #       collection_name=COLLECTION_NAME,
    #       query=seed_text,
    #       using="text",
    #       query_filter=Filter(must_not=[HasIdCondition(has_id=[point_id])]),
    #       limit=5,
    #       with_payload=True,
    #   )
    #
    # Step 3: Format and return results (use results.points, not results directly)
    #   return [
    #       {
    #           "point_id": str(r.id),
    #           "ticker": r.payload.get("ticker"),
    #           "chunk_text": r.payload.get("chunk_text"),
    #           "score": r.score,
    #           "quarter": r.payload.get("quarter"),
    #           "year": r.payload.get("year"),
    #       }
    #       for r in results.points
    #   ]

    return [{"error": "recommend_similar is not yet implemented — complete the TODOs!"}]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
