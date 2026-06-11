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
import logging
import os
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

# Worker liveness: the worker refreshes job["heartbeat"] at least every
# HEARTBEAT_INTERVAL_S; reconcile_jobs() treats anything older than
# STALE_HEARTBEAT_S (or a dead pid) as an orphaned job.
HEARTBEAT_INTERVAL_S = 30
STALE_HEARTBEAT_S = 120
INTERRUPTED_ERROR = (
    "interrupted (process restarted) — start the same call again to resume"
)

_lock = threading.Lock()

logger = logging.getLogger("jobs")
if not logging.getLogger().handlers and not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
logger.setLevel(logging.INFO)


# ── Job store ─────────────────────────────────────────────────────────────────


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _save(job: dict[str, Any], beat: bool = True) -> None:
    job["updated_at"] = _now_iso()
    if beat:
        job["heartbeat"] = job["updated_at"]
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    path = _job_path(job["job_id"])
    with _lock:
        # Preserve a cancel flag set by another process so the worker's
        # in-memory copy doesn't overwrite it back to False.
        if not job.get("cancel_requested") and path.exists():
            try:
                if json.loads(path.read_text()).get("cancel_requested"):
                    job["cancel_requested"] = True
            except (json.JSONDecodeError, OSError):
                pass
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job, indent=2))
        os.replace(tmp, path)


def load_job(job_id: str) -> dict[str, Any] | None:
    path = _job_path(job_id)
    return json.loads(path.read_text()) if path.exists() else None


def _read_all_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for p in sorted(JOBS_DIR.glob("*.json")):
        try:
            jobs.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return jobs


def load_jobs() -> list[dict[str, Any]]:
    if not JOBS_DIR.exists():
        return []
    reconcile_jobs()
    return sorted(_read_all_jobs(), key=lambda j: j.get("created_at", ""), reverse=True)


# ── Orphan detection ──────────────────────────────────────────────────────────


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def reconcile_jobs() -> list[dict[str, Any]]:
    """Self-heal orphaned jobs.

    Jobs run as daemon threads inside whichever process called start_job; if
    that process dies (e.g. `python app.py` is restarted) the job file stays
    status="running" forever. Any reader calling load_jobs() triggers this
    check: a pending/running job whose recorded worker pid is not alive, or
    whose heartbeat is stale, is marked failed so it can be restarted (and
    resume from on-disk artifacts). Returns the jobs that were healed.
    """
    healed: list[dict[str, Any]] = []
    if not JOBS_DIR.exists():
        return healed
    for job in _read_all_jobs():
        if job.get("status") not in ("pending", "running"):
            continue
        pid = job.get("pid")
        pid_dead = isinstance(pid, int) and not _pid_alive(pid)
        ts_raw = job.get("heartbeat") or job.get("updated_at") or job.get("created_at") or ""
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            ts = None
        age = (datetime.now(timezone.utc) - ts).total_seconds() if ts else None
        stale = age is None or age > STALE_HEARTBEAT_S
        if not (pid_dead or stale):
            continue
        reason = (
            f"worker pid {pid} is not alive"
            if pid_dead
            else f"no heartbeat for {int(age)}s (limit {STALE_HEARTBEAT_S}s)"
            if age is not None
            else "no heartbeat recorded"
        )
        job["status"] = "failed"
        job["error"] = INTERRUPTED_ERROR
        job.setdefault("log", []).append(f"[reconcile] {reason} — {INTERRUPTED_ERROR}")
        _save(job, beat=False)
        logger.info("[%s] reconciled orphaned job (%s) -> failed", job["job_id"], reason)
        healed.append(job)
    return healed


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
        logger.info("[%s] [%s] %s", job["job_id"], stage, note)
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
        # Resume path: recover metadata from the sidecar written by the
        # original download so later stages still know the audio file.
        sidecar_path = AUDIO_DIR / f"{stem}.json"
        if sidecar_path.exists():
            try:
                job["youtube_id"] = json.loads(sidecar_path.read_text()).get("youtube_id", "")
            except (json.JSONDecodeError, OSError):
                pass
        job["audio_file"] = out_path.name
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


def _retry(fn, attempts: int = 5, on_wait=None):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if "429" in str(exc) and attempt < attempts - 1:
                wait = 20 * (attempt + 1)
                if on_wait:
                    on_wait(wait, attempt + 1, attempts)
                time.sleep(wait)
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
    import importlib.util
    import os

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
        return False
    try:
        import pyannote.audio  # noqa: F401
    except Exception:
        return False
    # A transcription backend is also required, but it need not be
    # openai-whisper — faster-whisper is the default backend now.
    has_backend = (
        importlib.util.find_spec("faster_whisper") is not None
        or importlib.util.find_spec("whisper") is not None
    )
    return has_backend


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
        _progress(job, "transcribe", 0, 4,
                  "1/4 whisper transcription running — this can take tens of "
                  "minutes on CPU for a full call")
        t0 = time.monotonic()
        transcript["_words"] = pipeline.transcribe_file(audio_path)
        out_path.write_text(json.dumps(transcript, indent=2))
        logger.info("[%s] [transcribe] whisper done in %.0fs (%d words)",
                    job["job_id"], time.monotonic() - t0, len(transcript["_words"]))
    _progress(job, "transcribe", 1, 4,
              f"1/4 done — {len(transcript['_words'])} words")

    if _is_cancelled(job["job_id"]):
        raise InterruptedError("cancelled")
    if not transcript.get("chunks"):
        _progress(job, "transcribe", 1, 4, "2/4 speaker diarization (pyannote)")
        t0 = time.monotonic()
        segments = pipeline.diarize_audio(audio_path)
        logger.info("[%s] [transcribe] diarization done in %.0fs (%d segments)",
                    job["job_id"], time.monotonic() - t0, len(segments))
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


def _transcribe_gemini(
    job: dict[str, Any], audio_path: Path, out_path: Path,
    existing: dict[str, Any] | None = None,
) -> None:
    from google.genai import types
    from pydub import AudioSegment

    full = AudioSegment.from_file(str(audio_path))
    duration_s = len(full) / 1000.0
    n_chunks = max(1, int(duration_s // CHUNK_SECONDS) + (1 if duration_s % CHUNK_SECONDS > 1 else 0))

    # Resume: reuse chunks that already have text from a previous (partial) run.
    prior: dict[int, dict[str, Any]] = {}
    if existing:
        for c in existing.get("chunks", []):
            if str(c.get("text", "")).strip():
                prior[int(c.get("chunk_index", -1))] = c
    if prior:
        _progress(job, "transcribe", len(prior), n_chunks,
                  f"resuming — {len(prior)}/{n_chunks} clips already transcribed")

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
        if i in prior:
            chunks.append(prior[i])
            continue
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

        def on_wait(wait: int, attempt: int, attempts: int) -> None:
            _progress(job, "transcribe", i, n_chunks,
                      f"clip {i + 1}/{n_chunks} rate-limited (429) — waiting "
                      f"{wait}s before retry {attempt}/{attempts - 1}")

        response = _retry(call, on_wait=on_wait)
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
        # Persist partial progress (speakers=None marks it incomplete) so an
        # interrupted job resumes here instead of re-transcribing every clip.
        partial = {**_transcript_meta(job), "transcriber": "gemini-per-clip",
                   "speakers": None, "chunks": chunks}
        out_path.write_text(json.dumps(partial, indent=2))
        _progress(job, "transcribe", i + 1, n_chunks)
        if (i + 1) % 5 == 0 or i + 1 == n_chunks:
            logger.info("[%s] [transcribe] gemini clips %d/%d",
                        job["job_id"], i + 1, n_chunks)

    transcript = {**_transcript_meta(job), "transcriber": "gemini-per-clip",
                  "speakers": {}, "chunks": chunks}
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

    # A partial Gemini per-clip transcript must resume on the Gemini path —
    # its chunks are fixed 30s windows, not whisper/diarization output.
    gemini_partial = bool(existing and existing.get("transcriber") == "gemini-per-clip")

    if gemini_partial:
        _progress(job, "transcribe", 0, 1,
                  "resuming Gemini per-clip transcription from partial transcript")
    elif _diarization_available():
        try:
            _transcribe_whisper(job, audio_path, out_path, existing)
            return
        except InterruptedError:
            raise
        except Exception as exc:
            # Diarization IS available, so a failure here is transient
            # (model download, audio quirk, etc.). Falling back to Gemini
            # would write a `gemini-per-clip` transcript that permanently
            # pins this call to the Gemini path on every later resume — so
            # by default we fail loudly and keep the job retryable on the
            # pyannote path. Set DIARIZATION_FALLBACK=1 to opt into the old
            # silent Gemini degrade.
            if not os.environ.get("DIARIZATION_FALLBACK"):
                raise RuntimeError(
                    f"whisper/pyannote transcription failed "
                    f"({type(exc).__name__}: {exc}). Diarization is available, "
                    "so the job was not silently downgraded to Gemini — re-run "
                    "to retry pyannote, or set DIARIZATION_FALLBACK=1 to allow "
                    "the Gemini per-clip fallback."
                ) from exc
            _progress(job, "transcribe", 0, 1,
                      f"whisper/pyannote path failed ({type(exc).__name__}: {exc}) "
                      "— DIARIZATION_FALLBACK set, using Gemini per-clip transcription")
    else:
        _progress(job, "transcribe", 0, 1,
                  "whisper/pyannote/HF_TOKEN unavailable — using Gemini "
                  "per-clip transcription")
    _transcribe_gemini(job, audio_path, out_path,
                       existing if gemini_partial else None)


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

    logger.info("[%s] [embed] embedding %s", job["job_id"], transcript_path.name)

    def on_progress(done: int, total: int) -> None:
        if _is_cancelled(job["job_id"]):
            raise InterruptedError("cancelled")
        _progress(job, "embed", done, total)
        if done % EMBED_BATCH == 0 or done == total:
            logger.info("[%s] [embed] chunk %d/%d", job["job_id"], done, total)

    upserted = pipeline.process_transcript(
        transcript_path, client, cache, point_map, on_progress=on_progress
    )
    point_map_file.write_text(json.dumps(point_map, indent=2))
    _progress(job, "embed", 1, 1, f"upserted {upserted} points via ingest pipeline")
    return upserted


# ── Worker ────────────────────────────────────────────────────────────────────


def _heartbeat_loop(job_id: str, stop: threading.Event) -> None:
    """Refresh the job's heartbeat while the worker is inside long opaque
    steps (whisper transcription, diarization, 429 retry sleeps) so that
    reconcile_jobs() in other processes doesn't mistake it for an orphan.

    Only the timestamp fields are touched, under the same lock as _save, so a
    concurrent status/progress write can never be clobbered with stale data.
    """
    path = _job_path(job_id)
    while not stop.wait(HEARTBEAT_INTERVAL_S):
        with _lock:
            try:
                job = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if job.get("status") not in ("pending", "running"):
                return
            job["heartbeat"] = job["updated_at"] = _now_iso()
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(job, indent=2))
            os.replace(tmp, path)


def _run(job_id: str) -> None:
    job = load_job(job_id)
    if job is None:
        return
    started = time.monotonic()
    stop = threading.Event()
    threading.Thread(target=_heartbeat_loop, args=(job_id, stop), daemon=True).start()
    try:
        job["status"] = "running"
        job["pid"] = os.getpid()
        _save(job)

        logger.info("[%s] stage download", job_id)
        audio_path = _stage_download(job)
        logger.info("[%s] stage transcribe", job_id)
        _stage_transcribe(job, audio_path)
        logger.info("[%s] stage embed", job_id)
        points = _stage_embed(job)

        job = load_job(job_id) or job
        job["status"] = "done"
        job["percent"] = 100.0
        job["points_indexed"] = points
        job["log"].append(f"indexed {points} chunks into Qdrant")
        _save(job)
        logger.info("[%s] done in %.0fs — indexed %d points",
                    job_id, time.monotonic() - started, points)
    except InterruptedError:
        job = load_job(job_id) or job
        job["status"] = "cancelled"
        _save(job)
        logger.info("[%s] cancelled after %.0fs", job_id, time.monotonic() - started)
    except Exception as exc:
        job = load_job(job_id) or job
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        _save(job)
        logger.info("[%s] failed after %.0fs — %s", job_id,
                    time.monotonic() - started, job["error"])
    finally:
        stop.set()


def start_job(
    youtube_url: str, ticker: str, company: str, quarter: str, year: int, date: str
) -> dict[str, Any]:
    """Create a job record and launch the pipeline in a daemon thread.

    load_jobs() reconciles orphaned jobs first, so a job left "running" by a
    dead process won't block a restart; failed/cancelled jobs for the same
    call get a fresh job that resumes from on-disk artifacts (downloaded mp3,
    partial transcript, idempotent embed upserts).
    """
    # Reuse a genuinely active job for the same call instead of double-running.
    for existing in load_jobs():
        if (
            existing["ticker"] == ticker
            and existing["quarter"] == quarter
            and existing["year"] == year
            and existing["status"] in ("pending", "running")
        ):
            logger.info("[%s] reusing active job for %s %s %s",
                        existing["job_id"], ticker, quarter, year)
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
        "pid": os.getpid(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save(job)
    logger.info("[%s] start %s %s %s url=%s", job["job_id"], ticker, quarter, year,
                youtube_url)
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
