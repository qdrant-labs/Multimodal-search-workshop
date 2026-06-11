"""
Embedding helper with local cache fallback.

Used by both the ingestion pipeline and the MCP server so that the server
can answer queries without a live Gemini API key as long as the query text
was already cached during ingestion.

Model: gemini-embedding-2 (multimodal — text and audio share a 3072-dim
space). The ingestion pipeline embeds each chunk twice (text + audio) and
stores them as named vectors {text, audio}. At query time the server
embeds the user's text question with the same model and searches against
the `text` named vector (using='text').
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

CACHE_FILE = Path(__file__).parent.parent / "data" / "embedding_cache_v2.json"


EMBEDDING_MODEL = "models/gemini-embedding-2"

# Process-wide in-memory cache. The on-disk file holds ~1.1k × 3072-dim
# vectors (tens of MB), so parsing it on every embed_query() call adds
# hundreds of ms to each search. Load it once, lazily, and reuse it.
_cache: Optional[dict[str, list[float]]] = None


def embed_query(text: str, api_key: Optional[str] = None) -> list[float]:
    """
    Embed *text* using Gemini gemini-embedding-2 (3072 dimensions).

    The returned vector lives in the same shared multimodal space as the
    audio embeddings produced by the ingestion pipeline, so this single
    text vector can be searched against either the `text` or `audio`
    named vector in Qdrant.

    Falls back to the local embedding cache if the API is unavailable or
    the key is not set. Raises RuntimeError if both fail.
    """
    cache_key = "text:" + hashlib.sha256(text.encode()).hexdigest()
    cache = _get_cache()

    if cache_key in cache:
        return cache[cache_key]

    try:
        from google import genai  # type: ignore

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            raise EnvironmentError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=resolved_key)
        result = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
        vec: list[float] = list(result.embeddings[0].values)
        cache[cache_key] = vec
        _save_cache(cache)
        return vec
    except Exception as exc:
        raise RuntimeError(f"Embedding failed and no cache hit: {exc}") from exc


def _get_cache() -> dict[str, list[float]]:
    """Return the process-wide cache, loading it from disk on first use."""
    global _cache
    if _cache is None:
        if CACHE_FILE.exists():
            _cache = json.loads(CACHE_FILE.read_text())
        else:
            _cache = {}
    return _cache


def _save_cache(cache: dict[str, list[float]]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache))
