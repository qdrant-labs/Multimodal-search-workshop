"""Smoke test for the completed MCP server tools (runs against the live cluster)."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server.server import (
    get_audio_clip,
    get_news_context,
    recommend_similar,
    search_earnings,
)


def show(label: str, value: object, t0: float) -> None:
    print(f"\n=== {label} ({time.time() - t0:.2f}s) ===")
    print(json.dumps(value, indent=2, default=str)[:1500])


t0 = time.time()
results = search_earnings("data center demand outlook", ticker="NVDA")
show("search_earnings (NVDA)", results, t0)
assert results and "error" not in results[0], results

point_id = results[0]["point_id"]

t0 = time.time()
ranged = search_earnings(
    "tariffs and supply chain", date_range="2025-01-01:2025-12-31"
)
show("search_earnings (date_range 2025)", [
    {k: r.get(k) for k in ("ticker", "year", "score")} for r in ranged
], t0)
assert ranged and "error" not in ranged[0], ranged
assert all(r["year"] == 2025 for r in ranged), "date filter leaked other years"

t0 = time.time()
boosted = search_earnings("AI infrastructure investment", boost_recency=True)
show("search_earnings (boost_recency)", [
    {k: r.get(k) for k in ("ticker", "year", "score")} for r in boosted
], t0)
assert boosted and "error" not in boosted[0], boosted

t0 = time.time()
clip = get_audio_clip(point_id)
audio_len = len(clip.get("audio_base64", ""))
clip_preview = {**clip, "audio_base64": f"<{audio_len} b64 chars>"}
show("get_audio_clip", clip_preview, t0)
assert "error" not in clip and audio_len > 1000, clip_preview

t0 = time.time()
similar = recommend_similar(point_id)
show("recommend_similar", [
    {"ticker": r.get("ticker"), "score": r.get("score"),
     "text": (r.get("chunk_text") or "")[:80]} for r in similar
], t0)
assert similar and "error" not in similar[0], similar

t0 = time.time()
news = get_news_context(point_id)
news_preview = {**news, "articles": news.get("articles", [])[:2]}
show("get_news_context", news_preview, t0)
assert "error" not in news, news

print("\nAll smoke tests passed.")
