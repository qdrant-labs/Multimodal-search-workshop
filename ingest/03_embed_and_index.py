"""
Step 3: Embed transcript chunks (text + audio) and upsert into Qdrant.

For each chunk in each transcript JSON in data/transcripts/:
  - Slices the 30-second audio segment from data/audio/{audio_file} and
    saves it to data/audio_clips/{point_id}.mp3 (so the MCP server can
    serve it later without re-slicing)
  - Embeds the transcript text with Gemini gemini-embedding-2 (3072-dim)
  - Embeds the audio clip with the SAME multimodal model — text and audio
    share the embedding space, so a text query can rank audio neighbours
    and vice versa
  - Caches both vectors in data/embedding_cache_v2.json keyed by
    sha256(modality:content) to avoid recomputing on re-runs
  - (Re)creates the Qdrant collection 'earnings_calls' with named vectors
        text  → 3072-dim, cosine
        audio → 3072-dim, cosine
  - Upserts each chunk as a single point carrying BOTH named vectors

At search time the MCP server embeds the user's text query with the same
model and searches using='text'. To do audio→audio retrieval, search
using='audio'. Cross-modal retrieval works because the two named vectors
sit in the same shared space.

Usage:
    python3 ingest/03_embed_and_index.py
"""

import shutil as _shutil
import hashlib
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Ensure ffmpeg is on PATH (uses static binary if system ffmpeg is absent)
if not _shutil.which("ffmpeg"):
    try:
        import static_ffmpeg  # type: ignore
        static_ffmpeg.add_paths()
    except ImportError:
        pass

TRANSCRIPTS_DIR = Path("./data/transcripts")
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./data/audio"))
CLIPS_DIR = Path(os.getenv("CLIPS_DIR", "./data/audio_clips"))
CACHE_FILE = Path("./data/embedding_cache_v2.json")
POINT_MAP_FILE = TRANSCRIPTS_DIR / "point_map.json"

QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH: Optional[str] = os.getenv("QDRANT_PATH") or None
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "earnings_calls")

VECTOR_SIZE = 3072
EMBEDDING_MODEL = "models/gemini-embedding-2"
# Smaller batch because each point now carries two 3072-dim vectors
# (~50 KB per point serialised), and the qdrant-client default write
# timeout (5s) is too tight for the larger payload.
BATCH_SIZE = 10
QDRANT_TIMEOUT_S = 60


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def _load_cache() -> dict[str, list[float]]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache))


# ---------------------------------------------------------------------------
# Gemini multimodal embeddings
# ---------------------------------------------------------------------------

_GEMINI_CLIENT: Any = None


def _client() -> Any:
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        from google import genai  # type: ignore
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


def _embed(contents: Any, cache_key: str, cache: dict[str, list[float]]) -> list[float]:
    if cache_key in cache:
        return cache[cache_key]
    client = _client()
    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL, contents=contents)
            vec: list[float] = list(result.embeddings[0].values)
            cache[cache_key] = vec
            return vec
        except Exception as exc:
            if "429" in str(exc) and attempt < max_retries - 1:
                wait = 60 * (attempt + 1)
                for sec in range(wait, 0, -1):
                    print(
                        f"\r    [rate limit] retrying in {sec}s "
                        f"({attempt + 1}/{max_retries})...",
                        end="", flush=True,
                    )
                    time.sleep(1)
                print("\r" + " " * 60 + "\r", end="", flush=True)
            else:
                raise RuntimeError(f"Embedding failed: {exc}") from exc
    raise RuntimeError("Embedding failed after max retries")


def embed_text(text: str, cache: dict[str, list[float]]) -> list[float]:
    key = "text:" + hashlib.sha256(text.encode()).hexdigest()
    return _embed(text, key, cache)


def embed_audio(audio_bytes: bytes, cache: dict[str, list[float]]) -> list[float]:
    from google.genai import types  # type: ignore
    key = "audio:" + hashlib.sha256(audio_bytes).hexdigest()
    if key in cache:
        return cache[key]
    part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")
    return _embed([part], key, cache)


# ---------------------------------------------------------------------------
# Audio slicing (cached per source file)
# ---------------------------------------------------------------------------

_FULL_AUDIO_CACHE: dict[str, Any] = {}


def _load_full_audio(path: Path) -> Any:
    from pydub import AudioSegment  # type: ignore
    key = str(path)
    if key not in _FULL_AUDIO_CACHE:
        _FULL_AUDIO_CACHE[key] = AudioSegment.from_file(str(path))
    return _FULL_AUDIO_CACHE[key]


def slice_audio_bytes(full_audio_path: Path, start_s: float, end_s: float) -> bytes:
    """Slice [start_s, end_s] out of *full_audio_path* and return mp3 bytes."""
    audio = _load_full_audio(full_audio_path)
    clip = audio[int(start_s * 1000): int(end_s * 1000)]
    buf = io.BytesIO()
    clip.export(buf, format="mp3")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def ensure_collection(client: Any) -> None:
    """Create or recreate the collection with named vectors {text, audio}.

    Uses a *filterable HNSW* graph: `payload_m` adds extra HNSW edges per
    indexed payload value so that heavily-filtered searches (e.g. by ticker
    or date) stay fast and recall-accurate instead of degrading toward a
    brute-force scan. The extra edges are only built for fields that have a
    payload index, and only for points inserted *after* the index exists —
    so we create the indexes immediately after the collection, before upsert.
    """
    from qdrant_client.models import (  # type: ignore
        Distance,
        HnswConfigDiff,
        PayloadSchemaType,
        VectorParams,
    )

    desired = {
        "text": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        "audio": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    }
    # Filterable HNSW: m = global edges, payload_m = extra per-payload edges.
    hnsw = HnswConfigDiff(m=16, payload_m=16)

    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        info = client.get_collection(COLLECTION_NAME)
        existing_vectors = info.config.params.vectors
        if not isinstance(existing_vectors, dict) or set(existing_vectors) != set(desired):
            print(
                f"  Schema mismatch — recreating collection '{COLLECTION_NAME}'")
            client.delete_collection(COLLECTION_NAME)
            client.create_collection(
                collection_name=COLLECTION_NAME, vectors_config=desired, hnsw_config=hnsw
            )
        else:
            print(f"  Collection '{COLLECTION_NAME}' already exists (named: text, audio)")
            # Ensure filterable-HNSW edges are enabled on a pre-existing collection
            client.update_collection(COLLECTION_NAME, hnsw_config=hnsw)
    else:
        client.create_collection(
            collection_name=COLLECTION_NAME, vectors_config=desired, hnsw_config=hnsw
        )
        print(f"  Created collection '{COLLECTION_NAME}' (named: text, audio; filterable HNSW)")

    # `date` is indexed as DATETIME (not KEYWORD) so it supports both range
    # filtering and time-decay score boosting at query time.
    for field, schema in [
        ("ticker", PayloadSchemaType.KEYWORD),
        ("date", PayloadSchemaType.DATETIME),
        ("year", PayloadSchemaType.INTEGER),
        ("speaker", PayloadSchemaType.KEYWORD),
    ]:
        try:
            client.create_payload_index(COLLECTION_NAME, field, schema)
        except Exception:
            pass  # index already exists


# ---------------------------------------------------------------------------
# Per-transcript processing
# ---------------------------------------------------------------------------

def process_transcript(
    transcript_path: Path,
    client: Any,
    cache: dict[str, list[float]],
    point_map: dict[str, Any],
    on_progress: Any = None,
) -> int:
    """Embed + upsert one transcript. `on_progress(done, total)` is invoked
    per chunk so callers (e.g. background indexing jobs) can track progress."""
    from qdrant_client.models import PointStruct  # type: ignore

    data = json.loads(transcript_path.read_text())
    chunks: list[dict[str, Any]] = data.get("chunks", [])
    if not chunks:
        print(f"  [warn] {transcript_path.name} has no chunks, skipping")
        return 0

    ticker = data.get("ticker", "UNKNOWN")
    company = data.get("company", "")
    quarter = data.get("quarter", "")
    year = data.get("year", 0)
    date = data.get("date", "")
    audio_file = data.get("audio_file", "")
    youtube_id = data.get("youtube_id", "")

    full_audio_path = AUDIO_DIR / audio_file
    if not full_audio_path.exists():
        print(f"  [error] audio file missing: {full_audio_path}; skipping")
        return 0

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    batch: list[PointStruct] = []
    total = 0

    for chunk_no, chunk in enumerate(tqdm(chunks, desc=f"  {ticker}", unit="chunk"), start=1):
        if on_progress:
            on_progress(chunk_no, len(chunks))
        text = chunk.get("text", "").strip()
        if not text:
            continue

        stable_key = f"{ticker}_{quarter}_{year}_{chunk['chunk_index']}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, stable_key))

        start_s = float(chunk.get("start", 0.0))
        end_s = float(chunk.get("end", start_s))

        clip_path = CLIPS_DIR / f"{point_id}.mp3"
        if clip_path.exists():
            audio_bytes = clip_path.read_bytes()
        else:
            audio_bytes = slice_audio_bytes(full_audio_path, start_s, end_s)
            clip_path.write_bytes(audio_bytes)

        try:
            text_vec = embed_text(text, cache)
            audio_vec = embed_audio(audio_bytes, cache)
        except RuntimeError as exc:
            tqdm.write(f"    [error] chunk {chunk['chunk_index']}: {exc}")
            continue

        payload: dict[str, Any] = {
            "ticker": ticker,
            "company": company,
            "quarter": quarter,
            "year": year,
            "chunk_index": chunk["chunk_index"],
            "chunk_text": text,
            "speaker": chunk.get("speaker", "unknown"),
            "start_time": start_s,
            "end_time": end_s,
            "audio_file": audio_file,
            "youtube_id": youtube_id,
            "date": date,
        }

        batch.append(
            PointStruct(
                id=point_id,
                vector={"text": text_vec, "audio": audio_vec},
                payload=payload,
            )
        )
        point_map[point_id] = {
            "audio_file": audio_file,
            "start_time": start_s,
            "end_time": end_s,
        }

        if len(batch) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            total += len(batch)
            batch = []
            _save_cache(cache)

    if batch:
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        total += len(batch)
    _save_cache(cache)

    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except ImportError:
        print("ERROR: qdrant-client is not installed.  Run: pip install qdrant-client")
        sys.exit(1)

    transcript_files = [
        f for f in TRANSCRIPTS_DIR.glob("*.json") if f.name != "point_map.json"
    ]
    if not transcript_files:
        print(f"No transcript JSONs found in {TRANSCRIPTS_DIR.resolve()}")
        print("Run 02_transcribe_and_diarize.py first.")
        sys.exit(0)

    if QDRANT_URL:
        print(f"Connecting to Qdrant at {QDRANT_URL} ...")
        client = QdrantClient(
            url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT_S
        )
    else:
        path = QDRANT_PATH or "./data/qdrant_storage"
        print(f"Using local Qdrant at {path} ...")
        client = QdrantClient(path=path)
    ensure_collection(client)

    cache = _load_cache()
    point_map: dict[str, Any] = {}
    if POINT_MAP_FILE.exists():
        point_map = json.loads(POINT_MAP_FILE.read_text())

    grand_total = 0
    for transcript_path in sorted(transcript_files):
        print(f"\nProcessing {transcript_path.name} ...")
        n = process_transcript(transcript_path, client, cache, point_map)
        print(f"  Upserted {n} points (text + audio vectors each)")
        grand_total += n

    POINT_MAP_FILE.write_text(json.dumps(point_map, indent=2))

    print(f"\nTotal points upserted: {grand_total}")
    print(f"Embedding cache:        {CACHE_FILE.resolve()}")
    print(f"Point map:              {POINT_MAP_FILE.resolve()}")
    print(f"Audio clips:            {CLIPS_DIR.resolve()}")
    print("\nDone.  Next step: python ingest/04_build_asknews_context.py")


if __name__ == "__main__":
    main()
