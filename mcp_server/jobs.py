"""
Background indexing jobs with per-stage progress.

A job takes a new earnings call (YouTube URL + metadata) through the full
pipeline against the LOCAL writable Qdrant:

    download    yt-dlp → data/audio/{stem}.mp3
    transcribe  Whisper + pyannote diarization + Gemini speaker identification
                (same pipeline as ingest/02); falls back to per-clip Gemini
                flash-lite transcription over fixed 30s windows when
                whisper/pyannote/HF_TOKEN are unavailable
    embed       gemini-embedding-2 text+audio vectors → Qdrant upsert

Job state is persisted as JSON under data/jobs/{job_id}.json so any process
(MCP server, web app, CLI) can observe progress. The worker runs in a
daemon thread inside the process that started the job.
"""

import io
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
JOBS_DIR = REPO_ROOT / "data" / "jobs"
AUDIO_DIR = REPO_ROOT / "data" / "audio"
CLIPS_DIR = REPO_ROOT / "data" / "audio_clips"
TRANSCRIPTS_DIR = REPO_ROOT / "data" / "transcripts"

CHUNK_SECONDS = 30
EMBED_BATCH = 10
TRANSCRIBE_MODEL = "models/gemini-3.1-flash-lite"

STAGES = ("download", "transcribe", "embed")

_lock = threading.Lock()


# ── Job store ─────────────────────────────────────────────────────────────────


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _save(job: dict[str, Any]) -> None:
    job["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        _job_path(job["job_id"]).write_text(json.dumps(job, indent=2))


def load_job(job_id: str) -> dict[str, Any] | None:
    path = _job_path(job_id)
    return json.loads(path.read_text()) if path.exists() else None


def load_jobs() -> list[dict[str, Any]]:
    if not JOBS_DIR.exists():
        return []
    jobs = [json.loads(p.read_text()) for p in sorted(JOBS_DIR.glob("*.json"))]
    return sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True)


def _progress(job: dict[str, Any], stage: str, done: int, total: int, note: str = "") -> None:
    job["stage"] = stage
    job["progress"][stage] = {"done": done, "total": total}
    pct_per_stage = 100 / len(STAGES)
    stage_idx = STAGES.index(stage)
    frac = (done / total) if total else 0.0
    job["percent"] = round(stage_idx * pct_per_stage + frac * pct_per_stage, 1)
    if note:
        job["log"].append(f"[{stage}] {note}")
        job["log"] = job["log"][-50:]
    _save(job)


def _is_cancelled(job_id: str) -> bool:
    current = load_job(job_id)
    return bool(current and current.get("cancel_requested"))


# ── Pipeline stages ───────────────────────────────────────────────────────────


def _stage_download(job: dict[str, Any]) -> Path:
    import shutil

    if not shutil.which("ffmpeg"):
        try:
            import static_ffmpeg

            static_ffmpeg.add_paths()
        except ImportError:
            pass

    import yt_dlp

    stem = f"{job['ticker'].lower()}_{job['quarter'].lower()}_{job['year']}"
    out_path = AUDIO_DIR / f"{stem}.mp3"
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        _progress(job, "download", 1, 1, f"{out_path.name} already downloaded")
        return out_path

    def hook(d: dict[str, Any]) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total:
                _progress(job, "download", done, total)

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(AUDIO_DIR / f"{stem}.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}
        ],
        "progress_hooks": [hook],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(job["youtube_url"], download=True)

    job["youtube_id"] = info.get("id", "")
    job["audio_file"] = out_path.name
    sidecar = {
        "ticker": job["ticker"],
        "company": job["company"],
        "quarter": job["quarter"],
        "year": job["year"],
        "date": job["date"],
        "youtube_id": job["youtube_id"],
        "title": info.get("title", ""),
        "duration": info.get("duration", 0),
        "audio_file": out_path.name,
    }
    (AUDIO_DIR / f"{stem}.json").write_text(json.dumps(sidecar, indent=2))
    _progress(job, "download", 1, 1, f"saved {out_path.name} ({info.get('duration', 0)}s)")
    return out_path


def _gemini():
    import os

    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def _retry(fn, attempts: int = 5):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if "429" in str(exc) and attempt < attempts - 1:
                time.sleep(20 * (attempt + 1))
                continue
            raise


def _load_diarize_pipeline():
    """Import ingest/02_transcribe_and_diarize.py (module name starts with a
    digit, so it needs importlib) — the SAME whisper + pyannote + Gemini
    speaker-identification pipeline used to build the original dataset."""
    import importlib.util

    path = REPO_ROOT / "ingest" / "02_transcribe_and_diarize.py"
    spec = importlib.util.spec_from_file_location("transcribe_and_diarize", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diarization_available() -> bool:
    import os

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
        return False
    try:
        import pyannote.audio  # noqa: F401
        import whisper  # noqa: F401
    except Exception:
        return False
    return True


def _transcript_meta(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": job["ticker"],
        "company": job["company"],
        "quarter": job["quarter"],
        "year": job["year"],
        "date": job["date"],
        "audio_file": job.get("audio_file", ""),
        "youtube_id": job.get("youtube_id", ""),
    }


def _transcribe_whisper(
    job: dict[str, Any], audio_path: Path, out_path: Path,
    existing: dict[str, Any] | None,
) -> None:
    pipeline = _load_diarize_pipeline()

    transcript = existing or {**_transcript_meta(job), "speakers": None,
                              "_words": [], "chunks": []}

    if _is_cancelled(job["job_id"]):
        raise InterruptedError("cancelled")
    if not transcript.get("_words"):
        _progress(job, "transcribe", 0, 4, "1/4 whisper transcription")
        transcript["_words"] = pipeline.transcribe_file(audio_path)
        out_path.write_text(json.dumps(transcript, indent=2))
    _progress(job, "transcribe", 1, 4,
              f"1/4 done — {len(transcript['_words'])} words")

    if _is_cancelled(job["job_id"]):
        raise InterruptedError("cancelled")
    if not transcript.get("chunks"):
        _progress(job, "transcribe", 1, 4, "2/4 speaker diarization (pyannote)")
        segments = pipeline.diarize_audio(audio_path)
        if _is_cancelled(job["job_id"]):
            raise InterruptedError("cancelled")
        _progress(job, "transcribe", 2, 4, "3/4 building speaker-boundary chunks")
        transcript["chunks"] = pipeline.build_chunks_from_diarization(
            transcript["_words"], segments
        )
        out_path.write_text(json.dumps(transcript, indent=2))
    _progress(job, "transcribe", 3, 4,
              f"3/4 done — {len(transcript['chunks'])} chunks")

    if _is_cancelled(job["job_id"]):
        raise InterruptedError("cancelled")
    _progress(job, "transcribe", 3, 4, "4/4 speaker identification (Gemini)")
    chunks, speakers = pipeline.identify_speakers(transcript["chunks"], [{}])
    transcript["chunks"] = chunks
    transcript["speakers"] = speakers
    out_path.write_text(json.dumps(transcript, indent=2))
    _progress(job, "transcribe", 4, 4,
              f"transcript saved ({len(transcript['chunks'])} chunks, "
              f"{len(speakers)} speaker(s))")


def _transcribe_gemini(job: dict[str, Any], audio_path: Path, out_path: Path) -> None:
    from google.genai import types
    from pydub import AudioSegment

    full = AudioSegment.from_file(str(audio_path))
    duration_s = len(full) / 1000.0
    n_chunks = max(1, int(duration_s // CHUNK_SECONDS) + (1 if duration_s % CHUNK_SECONDS > 1 else 0))

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    client = _gemini()
    prompt = (
        "Transcribe this earnings-call audio segment verbatim. "
        "If a speaker introduces themselves or is clearly identifiable, prefix "
        "with 'Speaker Name:'. Return ONLY the transcript text, no commentary."
    )

    chunks: list[dict[str, Any]] = []
    for i in range(n_chunks):
        if _is_cancelled(job["job_id"]):
            raise InterruptedError("cancelled")
        start = i * CHUNK_SECONDS
        end = min((i + 1) * CHUNK_SECONDS, duration_s)
        stable_key = f"{job['ticker']}_{job['quarter']}_{job['year']}_{i}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, stable_key))
        clip_path = CLIPS_DIR / f"{point_id}.mp3"
        if not clip_path.exists():
            buf = io.BytesIO()
            full[int(start * 1000) : int(end * 1000)].export(buf, format="mp3")
            clip_path.write_bytes(buf.getvalue())
        audio_bytes = clip_path.read_bytes()
        part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")

        def call():
            return client.models.generate_content(
                model=TRANSCRIBE_MODEL, contents=[part, prompt]
            )

        response = _retry(call)
        text = (response.text or "").strip()
        speaker = "unknown"
        if ":" in text.split("\n", 1)[0][:60]:
            head = text.split(":", 1)[0].strip()
            if 0 < len(head) <= 40 and not head[0].isdigit():
                speaker = head
        chunks.append(
            {"chunk_index": i, "start": start, "end": end,
             "text": text, "speaker": speaker}
        )
        _progress(job, "transcribe", i + 1, n_chunks)

    transcript = {**_transcript_meta(job), "speakers": {}, "chunks": chunks}
    out_path.write_text(json.dumps(transcript, indent=2))
    _progress(job, "transcribe", n_chunks, n_chunks, "transcript saved")


def _stage_transcribe(job: dict[str, Any], audio_path: Path) -> None:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{job['ticker'].lower()}_{job['quarter'].lower()}_{job['year']}"
    out_path = TRANSCRIPTS_DIR / f"{stem}.json"

    existing: dict[str, Any] | None = (
        json.loads(out_path.read_text()) if out_path.exists() else None
    )
    if existing and existing.get("chunks") and existing.get("speakers") is not None:
        _progress(job, "transcribe", 1, 1,
                  f"{out_path.name} already transcribed "
                  f"({len(existing['chunks'])} chunks)")
        return

    if _diarization_available():
        try:
            _transcribe_whisper(job, audio_path, out_path, existing)
            return
        except InterruptedError:
            raise
        except Exception as exc:
            _progress(job, "transcribe", 0, 1,
                      f"whisper/pyannote path failed ({type(exc).__name__}: {exc}) "
                      "— falling back to Gemini per-clip transcription")
    else:
        _progress(job, "transcribe", 0, 1,
                  "whisper/pyannote/HF_TOKEN unavailable — using Gemini "
                  "per-clip transcription")
    _transcribe_gemini(job, audio_path, out_path)


def _load_ingest_pipeline():
    """Import ingest/03_embed_and_index.py (module name starts with a digit,
    so it needs importlib) — the SAME pipeline used to build the original
    dataset, guaranteeing compatible point IDs, payloads, and vectors."""
    import importlib.util

    path = REPO_ROOT / "ingest" / "03_embed_and_index.py"
    spec = importlib.util.spec_from_file_location("embed_and_index", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_embed(job: dict[str, Any]) -> int:
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from mcp_server.server import _qdrant

    pipeline = _load_ingest_pipeline()
    client = _qdrant()

    # Same collection config + payload indexes as the original ingestion.
    pipeline.ensure_collection(client)

    stem = f"{job['ticker'].lower()}_{job['quarter'].lower()}_{job['year']}"
    transcript_path = TRANSCRIPTS_DIR / f"{stem}.json"

    cache = pipeline._load_cache()
    point_map_file = TRANSCRIPTS_DIR / "point_map.json"
    point_map: dict[str, Any] = (
        json.loads(point_map_file.read_text()) if point_map_file.exists() else {}
    )

    def on_progress(done: int, total: int) -> None:
        if _is_cancelled(job["job_id"]):
            raise InterruptedError("cancelled")
        _progress(job, "embed", done, total)

    upserted = pipeline.process_transcript(
        transcript_path, client, cache, point_map, on_progress=on_progress
    )
    point_map_file.write_text(json.dumps(point_map, indent=2))
    _progress(job, "embed", 1, 1, f"upserted {upserted} points via ingest pipeline")
    return upserted


# ── Worker ────────────────────────────────────────────────────────────────────


def _run(job_id: str) -> None:
    job = load_job(job_id)
    if job is None:
        return
    try:
        job["status"] = "running"
        _save(job)

        audio_path = _stage_download(job)
        _stage_transcribe(job, audio_path)
        points = _stage_embed(job)

        job = load_job(job_id) or job
        job["status"] = "done"
        job["percent"] = 100.0
        job["points_indexed"] = points
        job["log"].append(f"indexed {points} chunks into Qdrant")
        _save(job)
    except InterruptedError:
        job = load_job(job_id) or job
        job["status"] = "cancelled"
        _save(job)
    except Exception as exc:
        job = load_job(job_id) or job
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        _save(job)


def start_job(
    youtube_url: str, ticker: str, company: str, quarter: str, year: int, date: str
) -> dict[str, Any]:
    """Create a job record and launch the pipeline in a daemon thread."""
    # Reuse an active job for the same call instead of double-running.
    for existing in load_jobs():
        if (
            existing["ticker"] == ticker
            and existing["quarter"] == quarter
            and existing["year"] == year
            and existing["status"] in ("pending", "running")
        ):
            return existing

    job: dict[str, Any] = {
        "job_id": uuid.uuid4().hex[:12],
        "youtube_url": youtube_url,
        "ticker": ticker,
        "company": company,
        "quarter": quarter,
        "year": year,
        "date": date,
        "status": "pending",
        "stage": None,
        "percent": 0.0,
        "progress": {},
        "log": [],
        "error": None,
        "cancel_requested": False,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save(job)
    threading.Thread(target=_run, args=(job["job_id"],), daemon=True).start()
    return job


def cancel_job(job_id: str) -> dict[str, Any] | None:
    job = load_job(job_id)
    if job is None:
        return None
    if job["status"] in ("pending", "running"):
        job["cancel_requested"] = True
        _save(job)
    return job
