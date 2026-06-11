"""
Step 02: Transcribe downloaded audio files with Whisper, diarize speaker
turns using pyannote.audio, then identify named speakers with Gemini.

For each .mp3 in data/audio/ (that has a matching .json sidecar) the script:
  - Loads the configured Whisper model (TRANSCRIBE_BACKEND/WHISPER_MODEL,
    default faster-whisper "base")
  - Transcribes with word-level timestamps enabled
  - Runs pyannote.audio speaker diarization on the audio to get speaker segments
  - Assigns each word to a speaker segment, then groups words into ~30-second chunks
  - Groups chunks into larger paragraphs for Gemini context
  - Sends each paragraph to Gemini Flash-Lite (IDENTIFICATION_MODEL) to resolve
    speaker labels to real names (e.g. "Jensen Huang") — falls back to
    "Speaker 0", "Speaker 1", etc.
  - Saves a structured JSON to data/transcripts/{ticker}_{quarter}_{year}.json

Requirements:
  - HF_TOKEN env var: required to download pyannote models.
    Accept the license for the DIARIZATION_MODEL pipeline (default
    pyannote/speaker-diarization-community-1, falls back to 3.1) at:
    https://huggingface.co/pyannote/speaker-diarization-community-1
  - GEMINI_API_KEY env var: required for speaker name identification.
  - pip install pyannote.audio

Output format:
  {
    "ticker": "NVDA",
    "company": "NVIDIA Corporation",
    "quarter": "Q3",
    "year": 2024,
    "date": "2023-11-21",
    "audio_file": "nvda_q3_2024.mp3",
    "youtube_id": "qNpyWGj-Kro",
    "speakers": {
      "Speaker 0": "Jensen Huang",
      "Speaker 1": "Colette Kress"
    },
    "chunks": [
      {
        "chunk_index": 0,
        "text": "...",
        "start": 0.0,
        "end": 30.1,
        "speaker": "Speaker 0"
      },
      ...
    ]
  }

Usage:
    python3 ingest/02_transcribe_and_diarize.py
"""

import shutil as _shutil
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

import warnings
warnings.filterwarnings("ignore", module=r"pyannote\.audio\.core\.io")

# Let unsupported MPS (Apple GPU) ops silently fall back to CPU instead of
# crashing. Must be set BEFORE torch is imported anywhere, so we do it at
# module import time (torch itself is imported lazily inside the functions).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

load_dotenv()

# Ensure ffmpeg is on PATH (uses static binary if system ffmpeg is absent)
if not _shutil.which("ffmpeg"):
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except ImportError:
        pass

AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./data/audio"))
TRANSCRIPT_DIR = Path(os.getenv("TRANSCRIPT_DIR", "./data/transcripts"))

CHUNK_DURATION = 30.0  # seconds per chunk

# Speaker identification model
IDENTIFICATION_MODEL = "gemini-3.1-flash-lite"
IDENTIFY_TOKENS_PER_PARAGRAPH = 2048

# Whisper config (env-overridable). Default model stays "base"; a smaller
# model (e.g. "tiny"/"small") is the other speed lever. WHISPER_DEVICE can
# force "cuda"/"mps"/"cpu"; otherwise the best available device is auto-picked.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE")  # None → auto-select

# Transcription backend selector. "faster-whisper" (CTranslate2) is the default:
# it's ~3-5x faster on CPU (int8) than openai-whisper at the same accuracy, with
# word timestamps. "openai-whisper" keeps the reference implementation. If
# faster-whisper is selected but can't be imported/instantiated, transcribe_file
# falls back to openai-whisper automatically.
TRANSCRIBE_BACKEND = os.getenv("TRANSCRIBE_BACKEND", "faster-whisper").strip().lower()

# Diarization model (env-overridable via DIARIZATION_MODEL). community-1 is the
# newest pyannote pipeline (best accuracy) and is the default; 3.1 is kept as a
# documented fallback that is tried automatically if the primary fails to load.
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
FALLBACK_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


_WHISPER_MPS_PATCHED = False


def _patch_whisper_mps_dtw() -> None:
    """Make Whisper's word-timestamp DTW step MPS-safe.

    whisper.timing.dtw() does ``x.double().cpu()`` — but ``.double()`` runs on
    the (MPS) tensor first, and MPS has no float64, raising
    "Cannot convert a MPS Tensor to float64". PYTORCH_ENABLE_MPS_FALLBACK can't
    intercept this because it's an explicit dtype cast, not an op-dispatch miss.
    We wrap dtw to move the tensor to CPU *before* the cast. Idempotent and a
    no-op for non-MPS tensors.
    """
    global _WHISPER_MPS_PATCHED
    if _WHISPER_MPS_PATCHED:
        return
    try:
        import whisper.timing as _wt

        _orig_dtw = _wt.dtw

        def _dtw_mps_safe(x: "Any") -> "Any":
            # Match whisper's dtw(x: Tensor) -> np.ndarray; typed as Any so we
            # don't need torch/numpy at import time.
            if getattr(getattr(x, "device", None), "type", None) == "mps":
                x = x.cpu()
            return _orig_dtw(x)

        # Intentional monkeypatch: our wrapper deliberately has a broader (Any)
        # signature than whisper's concrete dtw(Tensor) -> ndarray.
        _wt.dtw = _dtw_mps_safe  # ty: ignore[invalid-assignment]  # monkeypatch
        _WHISPER_MPS_PATCHED = True
    except Exception as exc:  # patch is best-effort
        print(f"  [warn] could not apply Whisper MPS dtw patch: {exc}")


def _best_torch_device(override: Optional[str] = None) -> str:
    """Pick the best available torch device, preferring cuda → mps → cpu.

    *override* (e.g. from WHISPER_DEVICE) forces a specific device when it is
    valid and actually available; otherwise we fall back to auto-selection.
    """
    import torch

    if override:
        choice = override.strip().lower()
        if choice == "cuda" and torch.cuda.is_available():
            return "cuda"
        if choice == "mps" and torch.backends.mps.is_available():
            return "mps"
        if choice == "cpu":
            return "cpu"
        print(f"  [warn] requested device '{override}' unavailable — auto-selecting")

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---------------------------------------------------------------------------
# Structured output schema for Gemini speaker identification
# ---------------------------------------------------------------------------

class SpeakerAssignment(BaseModel):
    """Speaker label and optional resolved name for one chunk."""
    speaker_label: str          # e.g. "Speaker 0"
    speaker_name: Optional[str] = None  # e.g. "Jensen Huang", None if unknown


class ParagraphDiarization(BaseModel):
    """Speaker identification result for one paragraph (group of chunks)."""
    assignments: list[SpeakerAssignment]


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

def transcribe_file(mp3_path: Path) -> list[dict[str, Any]]:
    """
    Transcribe a single mp3 and return a flat list of word dicts:
    ``[{word, start, end}]``.

    Dispatches on TRANSCRIBE_BACKEND ("faster-whisper" default | "openai-whisper").
    If faster-whisper is selected but its import/instantiation fails, we fall
    back to openai-whisper so environments without faster-whisper still work.
    The returned shape is identical across backends — jobs.py / ingest/03
    consume it unchanged.
    """
    if TRANSCRIBE_BACKEND == "openai-whisper":
        return _transcribe_openai_whisper(mp3_path)

    # Default: faster-whisper, with graceful fallback to openai-whisper.
    try:
        return _transcribe_faster_whisper(mp3_path)
    except Exception as exc:  # noqa: BLE001 — any import/instantiation failure
        print(
            f"  [warn] faster-whisper backend unavailable ({exc.__class__.__name__}: "
            f"{exc}) — falling back to openai-whisper")
        return _transcribe_openai_whisper(mp3_path)


def _faster_whisper_device_and_compute() -> tuple[str, str]:
    """Map WHISPER_DEVICE (or auto) to a (device, compute_type) for CTranslate2.

    faster-whisper has no MPS backend, so mps (and auto-selected mps) map to
    CPU int8 — the fastest CPU option and a no-accuracy-loss win on Apple
    Silicon per our benchmark. cuda uses float16; cpu uses int8.
    """
    device = _best_torch_device(WHISPER_DEVICE)
    if device == "cuda":
        return "cuda", "float16"
    if device == "mps":
        print("  [info] faster-whisper has no MPS backend — using device=cpu, "
              "compute_type=int8 (fastest on Apple Silicon)")
        return "cpu", "int8"
    return "cpu", "int8"


def _transcribe_faster_whisper(mp3_path: Path) -> list[dict[str, Any]]:
    """Transcribe with faster-whisper (CTranslate2). Same output shape as the
    openai-whisper path: a flat ``[{word, start, end}]`` list."""
    import time

    from faster_whisper import WhisperModel

    device, compute_type = _faster_whisper_device_and_compute()
    print(f"  Loading faster-whisper model '{WHISPER_MODEL}' "
          f"(device={device}, compute_type={compute_type}) ...")
    model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)

    print(f"  Transcribing {mp3_path.name} with faster-whisper ...")
    t0 = time.monotonic()
    segments, _info = model.transcribe(
        str(mp3_path),
        word_timestamps=True,
        language="en",
        condition_on_previous_text=False,  # prevents hallucination loops
        vad_filter=True,                   # drop non-speech regions
    )

    words: list[dict[str, Any]] = []
    for segment in segments:  # generator — iterating runs the transcription
        # Respect the same no-speech intent as the openai path where the info
        # is available (vad_filter already removes most silence).
        if getattr(segment, "no_speech_prob", 0.0) > 0.6:
            continue
        for w in (segment.words or []):
            words.append(
                {
                    # faster-whisper word objects expose .word/.start/.end;
                    # .word keeps the leading space like openai-whisper does.
                    "word": w.word,
                    "start": w.start if w.start is not None else 0.0,
                    "end": w.end if w.end is not None else 0.0,
                }
            )

    elapsed = time.monotonic() - t0
    print(f"  faster-whisper done in {elapsed:.1f}s "
          f"(device={device}, compute_type={compute_type}, {len(words)} words)")
    return words


def _transcribe_openai_whisper(mp3_path: Path) -> list[dict[str, Any]]:
    """Transcribe with openai-whisper (reference implementation). Returns a flat
    ``[{word, start, end}]`` list."""
    import time

    try:
        import whisper
    except ImportError:
        print("ERROR: openai-whisper is not installed.  Run: pip install openai-whisper")
        sys.exit(1)

    # Default openai-whisper to CPU when no device is forced: our benchmark
    # shows MPS is slower than CPU for whisper 'base'. WHISPER_DEVICE still wins.
    device = _best_torch_device(WHISPER_DEVICE or "cpu")
    if device == "mps":
        _patch_whisper_mps_dtw()
    print(f"  Loading Whisper model '{WHISPER_MODEL}' (device={device})...")
    # Load in fp32. Whisper on MPS has known issues with float16/sparse ops, so
    # we keep weights in fp32 there (and on CPU); fp16 is only safe on cuda.
    model = whisper.load_model(WHISPER_MODEL, device=device)

    # fp16 only on cuda; MPS/CPU run fp32 to avoid unsupported-op crashes.
    use_fp16 = device == "cuda"

    print(f"  Transcribing {mp3_path.name} (device={device}, fp16={use_fp16}) ...")
    t0 = time.monotonic()
    result = model.transcribe(
        str(mp3_path),
        word_timestamps=True,
        verbose=False,
        condition_on_previous_text=False,  # prevents hallucination loops
        no_speech_threshold=0.6,           # suppress segments likely to be silence
        logprob_threshold=-1.0,            # discard low-confidence segments
        language="en",
        fp16=use_fp16,
    )

    words: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        # Skip segments flagged as no-speech by Whisper
        if segment.get("no_speech_prob", 0.0) > 0.6:
            continue
        for w in segment.get("words", []):
            words.append(
                {
                    "word": w.get("word", ""),
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                }
            )

    elapsed = time.monotonic() - t0
    print(f"  openai-whisper done in {elapsed:.1f}s "
          f"(device={device}, {len(words)} words)")
    return words


# ---------------------------------------------------------------------------
# pyannote.audio diarization
# ---------------------------------------------------------------------------

def diarize_audio(mp3_path: Path) -> list[dict[str, Any]]:
    """
    Run pyannote.audio speaker diarization on *mp3_path*.

    Returns a list of segments sorted by start time:
        [{"start": float, "end": float, "speaker": str}, ...]

    Speaker labels are pyannote's internal IDs, e.g. "SPEAKER_00".

    Uses DIARIZATION_MODEL (default pyannote/speaker-diarization-community-1),
    falling back to pyannote/speaker-diarization-3.1 if the primary fails to
    load. Requires HF_TOKEN and accepted model license(s) at:
    https://huggingface.co/pyannote/speaker-diarization-community-1
    https://huggingface.co/pyannote/speaker-diarization-3.1
    """
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGINGFACE_TOKEN", "")
    if not hf_token:
        print(
            "  [warn] HF_TOKEN not set — skipping diarization, speakers will be unknown")
        return []

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("  [warn] pyannote.audio not installed — skipping diarization")
        print("         Run: pip install pyannote.audio")
        return []

    import torch

    # Try the configured model first (default: community-1, the newest/most
    # accurate pyannote pipeline), then fall back to 3.1 if it fails to load
    # (e.g. gated access not granted for the primary).
    primary = os.getenv("DIARIZATION_MODEL", DEFAULT_DIARIZATION_MODEL)
    candidates = list(dict.fromkeys([primary, FALLBACK_DIARIZATION_MODEL]))

    pipeline = None
    active_model = None
    for model_id in candidates:
        try:
            print(f"  Loading pyannote pipeline '{model_id}' ...")
            pipeline = Pipeline.from_pretrained(model_id, token=hf_token)
            active_model = model_id
            break
        except Exception as exc:  # noqa: BLE001 — try the next candidate
            print(f"  [warn] could not load '{model_id}': {exc}")

    if pipeline is None:
        print("  [warn] no diarization pipeline could be loaded — skipping diarization")
        return []

    # Move to the best device (cuda → mps → cpu). If moving to the GPU raises
    # (unsupported MPS op during init, etc.), fall back to CPU rather than
    # failing the whole job.
    device = _best_torch_device()
    try:
        pipeline = pipeline.to(torch.device(device))
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] could not move pipeline to {device} ({exc}) — using cpu")
        device = "cpu"
        pipeline = pipeline.to(torch.device("cpu"))

    print(f"  Running diarization with '{active_model}' on {mp3_path.name} "
          f"(device={device}) ...")

    import whisper as _whisper
    from tqdm import tqdm

    # whisper.load_audio pipes raw PCM from ffmpeg — same path that drove transcription
    print(f"  Loading audio for diarization ...")
    audio_np = _whisper.load_audio(str(mp3_path))  # float32, 16 kHz, mono
    audio_input = {"waveform": torch.tensor(
        audio_np).unsqueeze(0), "sample_rate": 16000}

    # Progress bar driven by pyannote's hook callback.
    # The hook is called per step (segmentation, embeddings, clustering);
    # we show one bar per step, updating as batches complete.
    bars: dict[str, tqdm] = {}

    def _hook(step_name: str, _chunk: Any, total: int = 1,
              completed: int = 0, **kwargs: Any) -> None:
        if step_name not in bars:
            bars[step_name] = tqdm(
                total=total,
                desc=f"    {step_name}",
                unit="batch",
                leave=True,
            )
        bar = bars[step_name]
        bar.total = total
        bar.n = completed
        bar.refresh()

    try:
        diarization = pipeline(audio_input, hook=_hook)
    finally:
        for bar in bars.values():
            bar.close()

    # DiarizeOutput wraps the Annotation; fall back gracefully for older versions
    annotation = getattr(diarization, "speaker_diarization", diarization)

    segments: list[dict[str, Any]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            {
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,  # e.g. "SPEAKER_00"
            }
        )

    # Normalise pyannote speaker IDs to "Speaker 0", "Speaker 1", ...
    seen: dict[str, str] = {}
    for seg in segments:
        raw = seg["speaker"]
        if raw not in seen:
            seen[raw] = f"Speaker {len(seen)}"
        seg["speaker"] = seen[raw]

    print(
        f"  Diarization found {len(seen)} speaker(s) across {len(segments)} segment(s)")
    return segments


# ---------------------------------------------------------------------------
# Word → speaker assignment and chunking
# ---------------------------------------------------------------------------

def _speaker_for_word(
    w_start: float, w_end: float, segments: list[dict[str, Any]]
) -> str:
    """
    Return the speaker label for a word by finding the diarization segment
    with maximum overlap with [w_start, w_end].
    Falls back to the nearest segment (by midpoint distance) if no overlap.
    """
    w_mid = (w_start + w_end) / 2
    best_speaker = "Speaker 0"
    best_overlap = -1.0
    best_dist = float("inf")

    for seg in segments:
        overlap = min(w_end, seg["end"]) - max(w_start, seg["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]
        if overlap <= 0:
            dist = min(abs(w_mid - seg["start"]), abs(w_mid - seg["end"]))
            if dist < best_dist:
                best_dist = dist
                if best_overlap <= 0:
                    best_speaker = seg["speaker"]

    return best_speaker


def build_chunks_from_diarization(
    words: list[dict[str, Any]],
    diarization_segments: list[dict[str, Any]],
    chunk_duration: float = CHUNK_DURATION,
) -> list[dict[str, Any]]:
    """
    Build chunks from words and pyannote diarization segments.

    Chunk boundaries are driven by speaker changes: a new chunk starts whenever
    the speaker changes. Long single-speaker turns are also split at
    chunk_duration to keep chunks a manageable size for Gemini.

    Each returned dict: chunk_index, text, start, end, speaker.
    """
    if not words:
        return []

    chunks: list[dict[str, Any]] = []
    current_words: list[str] = []
    chunk_start: float = words[0].get("start", 0.0)
    chunk_end: float = chunk_start
    current_speaker: str = _speaker_for_word(
        words[0].get("start", 0.0), words[0].get(
            "end", 0.0), diarization_segments
    )
    chunk_index = 0

    def flush(end: float) -> None:
        nonlocal chunk_index
        if not current_words:
            return
        chunks.append(
            {
                "chunk_index": chunk_index,
                "text": " ".join(w.strip() for w in current_words),
                "start": round(chunk_start, 3),
                "end": round(end, 3),
                "speaker": current_speaker,
            }
        )
        chunk_index += 1

    for word_info in words:
        word_text: str = word_info.get("word", "")
        w_start: float = word_info.get("start", chunk_end)
        w_end: float = word_info.get("end", w_start)
        spk = _speaker_for_word(w_start, w_end, diarization_segments)

        speaker_changed = spk != current_speaker
        too_long = current_words and (w_end - chunk_start >= chunk_duration)

        if speaker_changed or too_long:
            flush(chunk_end)
            current_words = []
            chunk_start = w_start
            current_speaker = spk

        current_words.append(word_text)
        chunk_end = w_end

    flush(chunk_end)
    return chunks


# ---------------------------------------------------------------------------
# Gemini speaker identification
# ---------------------------------------------------------------------------

def _try_identify_speakers(
    unknown_labels: list[str],
    label_texts: dict[str, str],
    known_speakers: dict[str, str],
    client: Any,
    context_chunks: list[dict[str, Any]] | None = None,
    participant_list: list[dict] = [{}],
) -> ParagraphDiarization:
    """
    Ask Gemini to identify real names for *unknown_labels* only.

    *label_texts* maps each unknown label to a sample of its speech.
    *known_speakers* maps already-resolved labels to names, for context.
    *context_chunks* are preceding chunks shown to Gemini as surrounding context
    (e.g. operator introductions just before an analyst speaks).
    """

    participants = [
        f'{p.get("name", "")} ({p.get("role", "")})' for p in participant_list]
    known_prts = ""
    if participants:
        participants = ", ".join(participants)
        known_prts = f"Known participants: {participants}.\n\n"

    known_ctx = ""
    if known_speakers:
        entries = ", ".join(f"{k} = {v}" for k, v in known_speakers.items())
        known_ctx = f"Already identified speakers: {entries}.\n\n"

    context_ctx = ""
    if context_chunks:
        lines = "\n".join(
            f"{c['speaker']}: {c['text']}" for c in context_chunks)
        context_ctx = f"Preceding context (for reference only):\n{lines}\n\n"

    speaker_samples = "\n".join(
        f"{label}: {label_texts[label]}" for label in unknown_labels
    )

    prompt = f"""You are a financial transcript analyst identifying speakers on an earnings call.

{known_prts}{known_ctx}{context_ctx}For each speaker label below, identify the real person's name 
if you can determine it from the transcript text or the preceding context (e.g. from introductions, 
self-references, or how others address them).

Rules:
- Only assign a name when you are confident from the text itself or the context above.
- Do NOT guess. Use null when unsure.
- Each label is a DIFFERENT person. Never assign the same name to two labels.
- Keep speaker_label exactly as shown.

Transcript samples:

{speaker_samples}
"""

    response = client.models.generate_content(
        model=IDENTIFICATION_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ParagraphDiarization,
        },
    )

    return response.parsed


def _split_into_paragraphs(
    chunks: list[dict[str, Any]],
    max_tokens: int = IDENTIFY_TOKENS_PER_PARAGRAPH,
) -> list[list[dict[str, Any]]]:
    """
    Group chunks into paragraphs whose rendered text stays under *max_tokens*.
    Uses tiktoken cl100k_base as a fast offline approximation for Gemini token counts.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def count_tokens(text): return len(enc.encode(text))
    except ImportError:
        def count_tokens(text): return len(text) // 4

    paragraphs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk["text"])
        if current and current_tokens + chunk_tokens > max_tokens:
            paragraphs.append(current)
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += chunk_tokens

    if current:
        paragraphs.append(current)

    return paragraphs


def identify_speakers(
    chunks: list[dict[str, Any]],
    participant_list: list[dict] = [{}],
    max_tokens: int = IDENTIFY_TOKENS_PER_PARAGRAPH,
) -> tuple[list[dict[str, Any]], dict[str, Optional[str]]]:
    """
    Run Gemini speaker identification over all chunks and return:
      - updated chunks list (speaker field replaced with resolved name or label)
      - speakers dict mapping label → resolved name (None if unresolved)
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("  [warn] GEMINI_API_KEY not set — skipping speaker identification")
        return chunks, {}

    try:
        from google import genai
    except ImportError:
        print("  [warn] google-genai not installed — skipping speaker identification")
        return chunks, {}

    client = genai.Client(api_key=api_key)

    speakers: dict[str, Optional[str]] = {c["speaker"]: None for c in chunks}
    paragraphs = _split_into_paragraphs(chunks, max_tokens)

    print(
        f"  Identifying speakers across {len(paragraphs)} paragraph(s) with "
        f"{IDENTIFICATION_MODEL} ...")

    for para_idx, para_chunks in enumerate(paragraphs):
        known = {k: v for k, v in speakers.items() if v is not None}
        unknown = [label for label in dict.fromkeys(c["speaker"] for c in para_chunks)
                   if speakers.get(label) is None]
        if not unknown:
            continue

        # Build a text sample per unknown label.
        # Include the last chunk of the previous paragraph as a prefix so that
        # introductions like "Our next question is from [Name]" are visible
        # even when the speaker's first chunk is at a paragraph boundary.
        prev_chunks = paragraphs[para_idx - 1] if para_idx > 0 else []
        context_chunks = prev_chunks[-1:] + para_chunks

        label_texts: dict[str, list[str]] = {label: [] for label in unknown}
        for chunk in context_chunks:
            label = chunk["speaker"]
            if label in label_texts:
                label_texts[label].append(chunk["text"])
        label_samples = {label: " ".join(
            texts[:3]) for label, texts in label_texts.items()}

        result: Optional[ParagraphDiarization] = None
        for attempt in range(3):
            try:
                result = _try_identify_speakers(
                    unknown, label_samples, known, client, prev_chunks[-3:], participant_list)
                break
            except Exception as exc:
                if attempt < 2 and ("503" in str(exc) or "429" in str(exc)):
                    wait = 10 * (attempt + 1)
                    print(
                        f"  [retry] Paragraph {para_idx} failed ({exc.__class__.__name__}), "
                        f"retrying in {wait}s ...")
                    import time
                    time.sleep(wait)
                else:
                    print(
                        f"  [warn] Speaker identification failed for paragraph {para_idx}: {exc}")
                    break
        if result is None:
            continue

        for assignment in result.assignments:
            label = assignment.speaker_label
            name = assignment.speaker_name
            if not name or label not in speakers:
                continue
            existing = speakers.get(label)
            if existing is None:
                speakers[label] = name
            else:
                # Prefer the more complete name — if one contains the other, keep the longer.
                # e.g. "Tim" -> "Tim Cook" wins; unrelated names keep the existing.
                if existing.lower() in name.lower():
                    speakers[label] = name

    unresolved = [label for label, name in speakers.items() if name is None]
    if unresolved:
        print(
            f"  [warn] Could not identify name(s) for: {', '.join(sorted(unresolved))}")

    updated_chunks = []
    for chunk in chunks:
        c = dict(chunk)
        label = c["speaker"]
        c["speaker"] = speakers[label] if speakers.get(
            label) is not None else label
        updated_chunks.append(c)

    return updated_chunks, speakers


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _base_transcript(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": meta["ticker"],
        "company": meta.get("company", ""),
        "quarter": meta["quarter"],
        "year": meta["year"],
        "date": meta.get("date", ""),
        "audio_file": meta["audio_file"],
        "youtube_id": meta.get("youtube_id", ""),
        "speakers": None,
        "_words": [],   # populated after Whisper, removed after diarization
        "chunks": [],   # populated after diarization
    }


def main() -> None:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    mp3_files = list(AUDIO_DIR.glob("*.mp3"))
    if not mp3_files:
        print(f"No .mp3 files found in {AUDIO_DIR.resolve()}")
        print("Run 01_download_audio.py or 01b_fetch_benzinga.py first.")
        sys.exit(0)

    print(f"Found {len(mp3_files)} audio file(s) in {AUDIO_DIR.resolve()}\n")

    for mp3_path in sorted(mp3_files):
        sidecar_path = mp3_path.with_suffix(".json")
        if not sidecar_path.exists():
            print(f"  [skip] {mp3_path.name} — no sidecar JSON found")
            continue

        meta = json.loads(sidecar_path.read_text())
        ticker = meta["ticker"]
        quarter = meta["quarter"]
        year = meta["year"]
        filename_stem = f"{ticker.replace('/', '-').lower()}_{quarter.lower()}_{year}"
        out_name = f"{filename_stem}.json"
        out_path = TRANSCRIPT_DIR / out_name

        existing: dict[str, Any] | None = None
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            if existing and existing.get("speakers") is not None:
                print(
                    f"  [skip] {out_name} already diarized ({len(existing['speakers'])} "
                    "speaker(s))")
                continue

        print(f"Processing {mp3_path.name} ...")
        try:
            transcript = existing or _base_transcript(meta)

            # Step 1: transcribe — save words immediately so we can resume if diarization fails
            if not transcript.get("_words"):
                transcript["_words"] = transcribe_file(mp3_path)
                out_path.write_text(json.dumps(transcript, indent=2))
                print(
                    f"  Saved {len(transcript['_words'])} words from transcription")
            else:
                print(
                    f"  [resume] {out_name} has {len(transcript['_words'])} words — "
                    "skipping transcription")

            # Step 2: diarize and build speaker-boundary chunks
            # _words is kept in the file so we never need to re-transcribe
            if not transcript.get("chunks"):
                diarization_segments = diarize_audio(mp3_path)
                transcript["chunks"] = build_chunks_from_diarization(
                    transcript["_words"], diarization_segments)
                out_path.write_text(json.dumps(transcript, indent=2))
                print(
                    f"  Diarization complete — {len(transcript['chunks'])} chunks")
            else:
                print(
                    f"  [resume] {out_name} has {len(transcript['chunks'])} chunks — "
                    "skipping diarization")

            # Step 3: speaker identification
            participant_list = transcript.get(
                "api_response", {}).get("participants", [{}])
            chunks, speakers = identify_speakers(
                transcript["chunks"], participant_list)
            transcript["chunks"] = chunks
            transcript["speakers"] = speakers
            out_path.write_text(json.dumps(transcript, indent=2))

            n_speakers = len(transcript["speakers"])
            print(
                f"  Saved {out_name} — {len(transcript['chunks'])} chunks, "
                f"{n_speakers} speaker(s): {transcript['speakers']}"
            )
        except Exception as exc:
            print(f"  [error] {mp3_path.name}: {exc}")

    print("\nDone.  Next step: python3 ingest/03_embed_and_index.py")


if __name__ == "__main__":
    main()
