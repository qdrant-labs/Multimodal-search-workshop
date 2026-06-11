#!/usr/bin/env bash
set -e

echo "============================================"
echo "  Berlin Workshop: Earnings Call MCP Server"
echo "============================================"
echo ""

# 1. Install Python dependencies
echo "[1/4] Installing Python dependencies..."
uv sync # if no uv install uv here: https://docs.astral.sh/uv/getting-started/installation/
echo "      Done."
echo ""

# 2. Install Playwright browser
echo "[2/4] Installing Playwright Chromium browser..."
uv run -m playwright install chromium
echo "      Done."
echo ""

# 3. Create data subdirectories
echo "[3/4] Creating data directories..."
mkdir -p data/audio
mkdir -p data/transcripts
mkdir -p data/audio_clips
mkdir -p data/asknews_cache
echo "      Done."
echo ""

# 4. Copy .env.example to .env if not present
echo "[4/4] Setting up environment file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "      Created .env from .env.example"
    echo "      IMPORTANT: Edit .env and fill in your API keys before running!"
else
    echo "      .env already exists, skipping."
fi
echo ""

echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your API keys:"
echo "       GEMINI_API_KEY   — from https://aistudio.google.com"
echo "       ASKNEWS_CLIENT_ID / ASKNEWS_CLIENT_SECRET — from https://asknews.app"
echo "       QDRANT_URL       — defaults to http://localhost:6333"
echo ""
echo "  2. Run the ingestion pipeline:"
echo "       run uv ingest/01_download_audio.py"
echo "       run uv ingest/02_transcribe.py"
echo "       run uv ingest/03_embed_and_index.py"
echo "       run uv ingest/04_cache_asknews.py"
echo ""
echo "  3. Register the MCP server with Claude:"
echo "       run uv cli/setup_mcp.py install"
echo ""
echo "  4. Open workshop/exercises.md and start building!"
echo ""
