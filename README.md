# Earnings Call MCP Server — Berlin Workshop

Build an **MCP (Model Context Protocol) server** that lets Claude search earnings call
transcripts by meaning, play back the exact audio moment, and see what was happening
in the news when management spoke.

---

## Demo

[![Watch the demo on YouTube](https://img.youtube.com/vi/R3icD5DWLHw/hqdefault.jpg)](https://youtu.be/R3icD5DWLHw)

▶️ **[Watch the demo on YouTube](https://youtu.be/R3icD5DWLHw)**

<details>
<summary>Or play the local recording</summary>

<video src="https://github.com/qdrant-labs/Multimodal-search-workshop/raw/main/docs/EarningsCalls.mp4" controls width="100%"></video>

> If the player doesn't load, [watch/download the demo video](docs/EarningsCalls.mp4).

</details>

---

## Architecture

```
INGESTION PIPELINE  (pre-built, run once)
═══════════════════════════════════════════

  Source A · YouTube ─► yt-dlp ─► MP3 ─► Whisper ─► transcript ───┐
                                        (word timestamps)         │   (Whisper only runs on
  Source B · Benzinga API ───────────────► transcript text + MP3  ┤    the YouTube path; Benzinga
                                                                  │    supplies the transcript)
                                                                  ▼
              pyannote.audio diarization  +  Gemini 2.5 Flash-Lite (resolve speaker names)
                                                                  │
                                                                  ▼
                          30-second chunks  { chunk_text , 30s audio clip }
                                     │                                  │
            ┌────────────────────────┘                                  └────────────────────────┐
            ▼                                                                                      ▼
  Gemini Embedding 2 ─► text vector (3072-d cosine)            Gemini Embedding 2 ─► audio vector (3072-d cosine)
            │            (embeds chunk_text)                    (embeds the audio clip DIRECTLY — no Whisper)    │
            └────────────────────────┐                                  ┌────────────────────────┘
                                     ▼                                  ▼
                        Qdrant · collection "earnings_calls"
                        named vectors { text, audio } in one shared multimodal space

  AskNews DeepNews ──► data/asknews_cache/{ticker}_{date}_{point_id}.json   (per-chunk world context)

                          ┌─────────────────────────────────────────────────────┐
                          │            Qdrant Cloud (vector database)           │
                          │  collection: earnings_calls                         │
                          │  named vectors: text, audio  (both 3072-dim cosine) │
                          │  payload: ticker · date · speaker · timestamps · …  │
                          └──────────────────────┬──────────────────────────────┘
                                                 │
                          ┌──────────────────────▼──────────────────────────────┐
                          │          MCP Server  mcp_server/server.py           │
                          │                                                      │
                          │  ► search_earnings(query, ticker?, date_range?)     │◄── YOU BUILD
                          │  ► get_audio_clip(point_id) → base64 MP3            │◄── YOU BUILD
                          │  ► get_news_context(point_id) → AskNews articles    │◄── YOU BUILD
                          │  ► recommend_similar(point_id) → similar chunks     │◄── YOU BUILD
                          └─────────┬──────────────────────────┬────────────────┘
                                    │                          │
              ┌─────────────────────▼──────┐    ┌─────────────▼────────────────┐
              │     Claude Desktop /       │    │    Web Demo  app.py          │
              │     Claude Code (CLI)      │    │    http://localhost:8000      │
              │  "Which CEOs mentioned     │    │  search box + audio players  │
              │   tariffs in Q1 2025?"     │    │  + news cards     │
              └────────────────────────────┘    └──────────────────────────────┘
```

**Data flow for a query:**
1. User types a natural-language question
2. MCP server embeds it with Gemini Embedding 2 (`models/gemini-embedding-2`, 3072 dims, multimodal)
3. Qdrant searches the `text` named vector (`using="text"`) for top-5 matches.
   Audio→audio search is also possible via `using="audio"` since both
   modalities share the same embedding space.
4. Server fetches a base64-encoded audio clip for each chunk (pre-sliced by the ingest pipeline)
5. Server uses [AskNews](https://asknews.app) [DeepNews](https://docs.asknews.app/en/deepnews) to find all relevant articles, tweets, google search results, and wikipedia pages from the last 5 years and up to the last 5 minutes.
6. Claude (or the web UI) presents chunks, playable audio, and world context together

---

## What You Build vs. What's Pre-built

| Component | Who builds it | Notes |
|---|---|---|
| Ingestion pipeline (`ingest/01–04`) | Pre-built | Run once to populate the DB |
| Qdrant collection | Pre-built via pipeline | 577 points across AAPL, AMZN, NVDA, TSLA — each carries named vectors `text` (3072-dim) and `audio` (3072-dim) |
| AskNews cache (`data/asknews_cache/`) | Pre-built via pipeline | News, tweets, and more for ticker+date+speaker per transcription chunk |
| Audio clips (`data/audio_clips/`) | Pre-built via pipeline | 30-second MP3 slices per point, also fed into the audio embedding |
| `mcp_server/embeddings.py` | Pre-built | Gemini Embedding 2 text-side embed + disk cache fallback |
| `mcp_server/server.py` — **`search_earnings`** | **You build** | Exercise 1 — core vector search |
| `mcp_server/server.py` — **`get_audio_clip`** | **You build** | Exercise 2 — retrieve + encode audio |
| `mcp_server/server.py` — **`get_news_context`** | **You build** | Exercise 3 — read AskNews cache |
| `mcp_server/server.py` — **`recommend_similar`** | **You build** | Exercise 4 (bonus) |
| `mcp_server/server_solution.py` | Reference only | Full working solution — peek if stuck |
| Web demo (`app.py`) | Pre-built | FastAPI app at localhost:8000 |
| CLI (`cli/setup_mcp.py`) | Pre-built | Registers server with Claude Desktop |
| Browser agent (`browser_agent/`) | Pre-built | Bonus: Playwright + SEC EDGAR |

### Recommended scope for a 90-minute workshop

- **Exercises 1–3** are the core and fit comfortably in 90 minutes.
- **Exercise 4** (`recommend_similar`) is a good stretch goal — it shows off Qdrant's
  nearest-neighbour API and only takes ~10 extra lines.
- The ingestion pipeline, web app, and browser agent are intentionally pre-built so
  participants can focus on the MCP/Qdrant interaction rather than boilerplate.

---

## Exercises

| # | Exercise | Tool / File | What it's about |
|---|----------|-------------|-----------------|
| 1 | `search_earnings` | `server.py` | The core tool. Embed a natural-language query, optionally filter by `ticker` and/or a `date_range`, run a vector search over the `text` named vector, and return matching transcript chunks. |
| 2 | `get_audio_clip` | `server.py` | Retrieve a point by ID, find its pre-sliced `.mp3` clip, base64-encode it, and return it with timestamps — or a clean error if missing. |
| 3 | `get_news_context` | `server.py` | For a given chunk, look up cached AskNews articles, tweets, and more around the chunk's context (with a live-fetch fallback). |
| 4 | `recommend_similar` *(stretch)* | `server.py` | "More like this": fetch a seed chunk's stored `text` vector, search with it, and exclude the seed itself (`HasIdCondition`). |
| 5 | Filterable HNSW + payload indexes *(stretch)* | `ingest/03_embed_and_index.py` | Conceptual — why the collection uses `payload_m=16` and indexes `date` as DATETIME, so heavily-filtered searches stay fast and accurate instead of degrading to a brute-force scan. Already applied to the cluster; you just verify it. |
| 6 | Time-based score boosting *(stretch)* | `server.py` | Add a `boost_recency` flag that reranks results with `score + 0.3 · exp_decay(now − date)` via prefetch + `FormulaQuery`, so recent calls surface higher. |
| Bonus | `get_sec_filings` | `server.py` + `browser_agent/sec_scraper.py` | Wire the Playwright SEC EDGAR scraper in as a 5th MCP tool returning 10-Q/10-K filings. |

See [`workshop/exercises.md`](workshop/exercises.md) for the overview and
[`workshop/implementation_guide.md`](workshop/implementation_guide.md) for
detailed, step-by-step instructions.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12 | `python3 --version`; use `uv` to install if needed |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) — free tier |
| `QDRANT_URL` + `QDRANT_API_KEY` | Qdrant Cloud cluster (pre-provisioned for workshop) |
| `ASKNEWS_API_KEY` | [AskNews](https://my.asknews.app) — if you want your own free month ($250 value) of AskNews, go to https://my.asknews.app/plans and use promo code `SEARCHWEEK` to get the Spelunker plan (it will ask for payment details, but your card will not be charged for your first month). Then create your API key in your settings at https://my.asknews.app/en/settings/api-credentials. Otherwise, a test key is available that will work for the duration of the workshop. |
| `HF_TOKEN` | Required for the diarization in step 2. Accept terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1), [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0), and [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1). |
| ffmpeg | Bundled via `static-ffmpeg` — no system install needed |

---

## Quick Start

```bash
# 1. Clone / open the project
cd BerlinWorkshop

# 2. Install dependencies
uv venv --python 3.12 && source .venv/bin/activate # if no uv install uv here: https://docs.astral.sh/uv/getting-started/installation/
python -m ensurepip && python -m pip install -r requirements.txt  # or: uv pip install -r requirements.txt

# 3. Copy env template and fill in your API keys
cp .env.example .env
nano .env   # set GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY

# 4. (Instructor only) Run the ingestion pipeline
python3 ingest/01_download_audio.py   # download earnings calls from YouTube
python3 ingest/01b_fetch_benzinga.py   # fetch transcripts and audio from Benzinga API (optional)
python3 ingest/02_transcribe_and_diarize.py # Whisper transcription → 30s chunks + pyannote diarization + Gemini speaker ID
python3 ingest/03_embed_and_index.py  # Gemini embeddings → Qdrant Cloud
python3 ingest/04_build_asknews_context.py                # pre-fetch news (optional)

# 5. Register the MCP server with Claude Desktop
python3 cli/setup_mcp.py install
python3 cli/setup_mcp.py status           # verify

# 6. Open the exercises and start building
open workshop/exercises.md

# 7. Run the web demo to verify your implementation
python3 app.py                            # http://localhost:8000
```

---

## Project Structure

```
BerlinWorkshop/
├── README.md                         ← you are here
├── requirements.txt
├── setup.sh
├── .env.example                      ← copy to .env and fill in keys
├── app.py                            ← FastAPI web demo (pre-built)
├── demo.py                           ← CLI demo with rich tables
├── data/
│   ├── audio/                        ← downloaded .mp3 files (AAPL, AMZN, NVDA, TSLA)
│   ├── transcripts/                  ← Whisper JSON chunks
│   ├── audio_clips/                  ← pre-sliced clips keyed by Qdrant point_id
│   ├── asknews_cache/                ← {TICKER}_{DATE}_{POINT_ID}.json, one per transcription chunk for each earnings call
│   └── embedding_cache.json          ← offline embedding fallback (sha256 keyed)
├── ingest/
│   ├── 01_download_audio.py          ← yt-dlp → MP3
│   ├── 01b_fetch_benzinga.py         ← api → MP3 + transcript
│   ├── 02_transcribe_and_diarize.py  ← Whisper transcription + pyannote diarization + Gemini speaker ID
│   ├── 03_embed_and_index.py         ← Gemini Embedding 2 (text + audio) → Qdrant upsert
│   └── 04_build_asknews_context.py   ← AskNews context → JSON cache
├── mcp_server/
│   ├── server.py                     ← SKELETON — participants complete this
│   ├── server_solution.py            ← full working solution (instructor reference)
│   └── embeddings.py                 ← embed_query() with disk cache fallback
├── browser_agent/
│   └── sec_scraper.py                ← Playwright + SEC EDGAR (bonus exercise)
├── cli/
│   └── setup_mcp.py                  ← installs server into Claude Desktop config
└── workshop/
    └── exercises.md                  ← step-by-step workshop guide
```

---

## Qdrant Payload Schema

Each point represents a ~30-second transcript chunk and carries TWO
named vectors in the same multimodal space produced by Gemini Embedding 2:

```json
{
  "id": "<uuid v5, stable per chunk>",
  "vector": {
    "text":  [3072 floats],   // gemini-embedding-2 over chunk_text
    "audio": [3072 floats]    // gemini-embedding-2 over the audio clip
  },
  "payload": {
    "ticker":      "TSLA",
    "company":     "Tesla Inc.",
    "quarter":     "Q1",
    "year":        2025,
    "chunk_index": 12,
    "chunk_text":  "We are navigating the tariff environment carefully...",
    "speaker":     "Elon Musk",
    "start_time":  342.1,
    "end_time":    372.8,
    "audio_file":  "tsla_q1_2025.mp3",
    "youtube_id":  "vs4cfyyMWhQ",
    "date":        "2025-04-22"
  }
}
```

Queries must specify which named vector to search (`using="text"` or
`using="audio"`). Because both vectors live in the same shared space, a
text query can rank against `audio` and vice versa.

**Payload indexes** (required for filtered search):

| Field | Type | Used by |
|---|---|---|
| `ticker` | KEYWORD | `search_earnings(ticker=...)` |
| `date` | KEYWORD | `search_earnings(date_range=...)` |
| `year` | INTEGER | range queries |

---

## Earnings Calls in the Dataset

| Ticker | Company | Quarter | Date | Duration |
|---|---|---|---|---|
| NVDA | NVIDIA Corporation | Q3 FY2024 | 2023-11-21 | 121 chunks |
| AAPL | Apple Inc. | Q2 FY2025 | 2025-04-30 | 121 chunks |
| AMZN | Amazon.com Inc. | Q1 2025 | 2025-05-01 | 161 chunks |
| TSLA | Tesla Inc. | Q1 2025 | 2025-04-22 | 174 chunks |

---

## Test Questions for Claude

Once the server is running and registered:

```
Which CEOs mentioned tariffs in Q1 2025 earnings calls?
What did NVDA say about data center demand in Q3 2024?
How did Apple describe the impact of tariffs on its supply chain?
What was Amazon's tone on consumer spending in Q1 2025?
Find all mentions of AI infrastructure investment
Compare how TSLA and AAPL described macroeconomic uncertainty
Play the audio for that last transcript chunk
Find other earnings chunks similar to that one
```

---

## Troubleshooting

**MCP server not appearing in Claude:**
```bash
python cli/setup_mcp.py install
# then restart Claude Desktop (Cmd+Q, reopen)
```

**Embedding errors:**
- Check `GEMINI_API_KEY` in `.env`
- The `data/embedding_cache_v2.json` provides offline fallback once populated.
  Old caches from the text-only `gemini-embedding-001` pipeline live in
  `data/embedding_cache.json` and are not reused (different vector space)

**Qdrant connection errors:**
- Check `QDRANT_URL` and `QDRANT_API_KEY` in `.env`
- For local fallback: set `QDRANT_PATH=./data/qdrant_storage` and leave `QDRANT_URL` empty

**Audio not playing:**
- Check that `data/audio_clips/` contains `.mp3` files
- If empty, `pydub` will slice on demand from `data/audio/` (needs `static-ffmpeg` installed)

**Whisper takes too long (if re-transcribing):**
- In `ingest/02_transcribe_and_diarize.py` change `whisper.load_model("base")` to `"tiny"` for speed
