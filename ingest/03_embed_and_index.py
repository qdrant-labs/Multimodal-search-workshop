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
import random
import sys
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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

# Embedding is I/O-bound on Gemini round-trips (two per chunk), so we fan the
# chunks out across a bounded thread pool. EMBED_CONCURRENCY caps the number of
# in-flight Gemini calls; the default of 8 is conservative enough to stay under
# typical free-tier limits while still giving a large speedup. Set it lower if
# the key is rate-limited, higher on a paid key.
EMBED_CONCURRENCY = max(1, int(os.getenv("EMBED_CONCURRENCY", "8") or "8"))

# Per-task exponential backoff (with jitter) for 429s. Unlike the old fixed
# 60s-per-attempt sleep, this scales per task so a single rate-limited worker
# doesn't pointlessly stall the whole pool.
_BACKOFF_BASE_S = float(os.getenv("EMBED_BACKOFF_BASE_S", "2") or "2")
_BACKOFF_CAP_S = float(os.getenv("EMBED_BACKOFF_CAP_S", "60") or "60")
_EMBED_MAX_RETRIES = 6


# ---------------------------------------------------------------------------
# Embedding cache (shared across worker threads — guarded by _CACHE_LOCK)
# ---------------------------------------------------------------------------

# The cache dict is read/written by every worker thread; plain dict mutation is
# not atomic enough to race on, and json.dumps over a dict being mutated in
# another thread raises "dictionary changed size during iteration". Both the
# in-memory access and the on-disk snapshot are therefore serialised here.
_CACHE_LOCK = threading.Lock()


def _load_cache() -> dict[str, list[float]]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _cache_get(cache: dict[str, list[float]], key: str) -> Optional[list[float]]:
    with _CACHE_LOCK:
        return cache.get(key)


def _cache_put(cache: dict[str, list[float]], key: str, vec: list[float]) -> None:
    with _CACHE_LOCK:
        cache[key] = vec


def _save_cache(cache: dict[str, list[float]]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Snapshot under the lock so we never serialise a dict that a worker is
    # concurrently mutating, then write the snapshot outside the lock.
    with _CACHE_LOCK:
        snapshot = dict(cache)
    CACHE_FILE.write_text(json.dumps(snapshot))


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
    cached = _cache_get(cache, cache_key)
    if cached is not None:
        return cached
    client = _client()
    for attempt in range(_EMBED_MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL, contents=contents)
            vec: list[float] = list(result.embeddings[0].values)
            _cache_put(cache, cache_key, vec)
            return vec
        except Exception as exc:
            if "429" in str(exc) and attempt < _EMBED_MAX_RETRIES - 1:
                # Concurrency-aware backoff: exponential growth capped at
                # _BACKOFF_CAP_S, plus per-task jitter so simultaneously
                # rate-limited workers don't retry in lockstep (and a single
                # worker's wait never blocks the others).
                delay = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2 ** attempt))
                delay += random.uniform(0, _BACKOFF_BASE_S)
                time.sleep(delay)
            else:
                raise RuntimeError(f"Embedding failed: {exc}") from exc
    raise RuntimeError("Embedding failed after max retries")


def embed_text(text: str, cache: dict[str, list[float]]) -> list[float]:
    key = "text:" + hashlib.sha256(text.encode()).hexdigest()
    return _embed(text, key, cache)


def embed_audio(audio_bytes: bytes, cache: dict[str, list[float]]) -> list[float]:
    from google.genai import types  # type: ignore
    key = "audio:" + hashlib.sha256(audio_bytes).hexdigest()
    cached = _cache_get(cache, key)
    if cached is not None:
        return cached
    part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")
    return _embed([part], key, cache)


# ---------------------------------------------------------------------------
# Audio slicing (cached per source file)
# ---------------------------------------------------------------------------

_FULL_AUDIO_CACHE: dict[str, Any] = {}
_AUDIO_CACHE_LOCK = threading.Lock()


def _load_full_audio(path: Path) -> Any:
    from pydub import AudioSegment  # type: ignore
    key = str(path)
    # Guard the shared decode cache so concurrent first-touches of the same
    # source file don't double-decode or race on the dict. Decoding happens
    # under the lock; once loaded, the AudioSegment is immutable and safe to
    # slice from many threads.
    with _AUDIO_CACHE_LOCK:
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

    # Decode the source audio once, up front, so the worker threads only ever
    # *read*/slice from the shared (now-immutable) AudioSegment — no concurrent
    # first-touch decode races inside the pool.
    _load_full_audio(full_audio_path)

    n_chunks = len(chunks)

    def _embed_chunk(chunk: dict[str, Any]) -> Optional[tuple[Any, str, dict[str, Any]]]:
        """Embed text + audio for one chunk and build its PointStruct.

        Runs on a worker thread. Returns (point, point_id, point_map_entry) or
        None when the chunk is empty or its embedding fails (the original
        sequential code skipped both cases too). All shared state it touches —
        the embedding cache and the audio-decode cache — is internally locked.
        """
        text = chunk.get("text", "").strip()
        if not text:
            return None

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
            return None

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
        point = PointStruct(
            id=point_id,
            vector={"text": text_vec, "audio": audio_vec},
            payload=payload,
        )
        entry = {
            "audio_file": audio_file,
            "start_time": start_s,
            "end_time": end_s,
        }
        return point, point_id, entry

    batch: list[PointStruct] = []
    total = 0
    done = 0
    progress = tqdm(total=n_chunks, desc=f"  {ticker}", unit="chunk")

    # Fan the chunks out across a bounded pool; each future does both Gemini
    # round-trips for one chunk, so at most EMBED_CONCURRENCY calls are ever
    # in flight. Results are consumed as they complete — upsert batching, cache
    # snapshots and point_map writes all stay on this (single) main thread, so
    # the output schema is byte-identical to the sequential path; only the
    # order in which points are batched changes, which Qdrant is agnostic to.
    cancelled = False
    executor = ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY)
    try:
        pending = {executor.submit(_embed_chunk, chunk) for chunk in chunks}
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                result = fut.result()
                done += 1
                progress.update(1)
                if result is not None:
                    point, point_id, entry = result
                    batch.append(point)
                    point_map[point_id] = entry
                    if len(batch) >= BATCH_SIZE:
                        client.upsert(collection_name=COLLECTION_NAME, points=batch)
                        total += len(batch)
                        batch = []
                        _save_cache(cache)
                if on_progress:
                    # on_progress drives the job UI and raises InterruptedError
                    # on cancellation — propagate it after tearing the pool down.
                    on_progress(done, n_chunks)
    except InterruptedError:
        cancelled = True
        for fut in pending:
            fut.cancel()
        raise
    finally:
        progress.close()
        # Don't wait for in-flight futures when cancelling — return promptly.
        executor.shutdown(wait=not cancelled, cancel_futures=cancelled)

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
