"""
Earnings Call Audio Search — web demo.

Run with:  .venv/bin/python app.py
Then open: http://localhost:8000
"""

import base64
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

from mcp_server.server_solution import search_earnings, get_audio_clip, get_news_context

app = FastAPI()

HTML_PAGE = (
    "<!doctype html><html lang='en'><head>"
    "<meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>Earnings Call Search</title>"
    "<style>"
    "*{box-sizing:border-box;margin:0;padding:0}"
    "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
    "background:#0f1117;color:#e0e0e0;min-height:100vh;padding:32px 16px}"
    ".wrap{max-width:860px;margin:0 auto}"
    "h1{font-size:1.6rem;font-weight:700;margin-bottom:4px;color:#fff}"
    ".sub{color:#888;font-size:.9rem;margin-bottom:28px}"
    ".search-row{display:flex;gap:10px;margin-bottom:32px}"
    "input[type=text]{flex:1;padding:12px 16px;border-radius:8px;border:1px solid #333;"
    "background:#1a1d27;color:#fff;font-size:1rem;outline:none}"
    "input[type=text]:focus{border-color:#5b6af0}"
    "button{padding:12px 24px;border-radius:8px;border:none;background:#5b6af0;"
    "color:#fff;font-size:1rem;font-weight:600;cursor:pointer;white-space:nowrap}"
    "button:hover{background:#6b7af8}"
    "button:disabled{opacity:.5;cursor:not-allowed}"
    ".spinner{display:none;margin:40px auto;text-align:center;color:#888}"
    ".results{display:flex;flex-direction:column;gap:20px}"
    ".card{background:#1a1d27;border:1px solid #2a2d3d;border-radius:12px;"
    "padding:20px;transition:border-color .2s}"
    ".card:hover{border-color:#5b6af0}"
    ".card-header{display:flex;align-items:center;gap:12px;margin-bottom:12px;"
    "flex-wrap:wrap}"
    ".badge{background:#5b6af0;color:#fff;font-size:.75rem;font-weight:700;"
    "padding:3px 10px;border-radius:20px;white-space:nowrap}"
    ".badge.aapl{background:#555}"
    ".badge.amzn{background:#e47911}"
    ".badge.wmt{background:#007dc6}"
    ".badge.tsla{background:#cc0000}"
    ".badge.nvda{background:#76b900}"
    ".meta{font-size:.82rem;color:#888}"
    ".score{margin-left:auto;font-size:.82rem;color:#aaa}"
    ".quote{font-size:.95rem;line-height:1.7;color:#ccc;margin-bottom:14px;"
    "border-left:3px solid #5b6af0;padding-left:12px;white-space:pre-wrap}"
    "audio{width:100%;border-radius:6px;margin-top:4px;accent-color:#5b6af0}"
    ".audio-label{font-size:.78rem;color:#666;margin-bottom:4px}"
    ".no-clip{font-size:.82rem;color:#555;font-style:italic}"
    ".news-section{margin-top:16px;border-top:1px solid #2a2d3d;padding-top:14px}"
    ".news-label{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:#888;"
    "text-transform:uppercase;margin-bottom:10px}"
    ".news-window{font-size:.75rem;color:#555;margin-left:8px;font-weight:400}"
    ".news-scroll{max-height:380px;overflow-y:auto;display:flex;flex-direction:column;"
    "gap:8px;padding-right:4px}"
    ".news-scroll::-webkit-scrollbar{width:4px}"
    ".news-scroll::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.2);border-radius:4px}"
    ".article{background:#13151f;border:0.5px solid #2a2d3d;border-radius:10px;"
    "display:flex;gap:0;overflow:hidden;transition:border-color .2s;flex-shrink:0;cursor:pointer}"
    ".article:hover{border-color:#5b6af0}"
    ".article.expanded{overflow:visible!important;align-items:flex-start}"
    ".article-summary.expanded{display:block!important;-webkit-line-clamp:unset;"
    "-webkit-box-orient:unset;overflow:visible!important;max-height:none}"
    ".article-image-wrap{width:80px;min-width:80px;height:80px;overflow:hidden;flex-shrink:0}"
    ".article-image{width:100%;height:100%;object-fit:cover;display:block}"
    ".article-image-placeholder{width:80px;min-width:80px;height:80px;"
    "background:#1a1d27;display:flex;align-items:center;justify-content:center;"
    "color:#444;font-size:20px;flex-shrink:0}"
    ".article-body{flex:1;padding:9px 12px;min-width:0;display:flex;flex-direction:column;gap:3px}"
    ".article-title{font-size:.88rem;font-weight:600;color:#ddd;line-height:1.4}"
    ".article-title a{color:#8b9cf8;text-decoration:none}"
    ".article-title a:hover{text-decoration:underline;color:#a8b8ff}"
    ".article-meta{display:flex;align-items:center;gap:6px;flex-wrap:wrap}"
    ".meta-source{font-size:.75rem;font-weight:600;color:#aaa}"
    ".meta-dot{color:#444;font-size:.75rem}"
    ".meta-date{font-size:.75rem;color:#666}"
    ".meta-lang{font-size:.7rem;color:#666;background:rgba(255,255,255,0.06);"
    "border:0.5px solid #333;border-radius:3px;padding:1px 4px;text-transform:uppercase}"
    ".article-summary{font-size:.8rem;color:#777;line-height:1.5;"
    "display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;cursor:pointer}"
    ".article-summary.expanded{display:block;-webkit-line-clamp:unset}"
    ".art-badges{display:flex;gap:4px;flex-wrap:wrap;margin-top:1px}"
    ".art-badge{font-size:.7rem;padding:1px 6px;border-radius:4px;font-weight:500}"
    ".art-badge-pos{background:rgba(74,180,74,0.15);color:#6dca6d}"
    ".art-badge-neg{background:rgba(220,60,60,0.15);color:#e07070}"
    ".art-badge-neu{background:rgba(255,255,255,0.06);color:#666}"
    ".art-badge-bias{background:rgba(120,100,220,0.15);color:#b0a0f0}"
    ".art-badge-type{background:rgba(40,180,160,0.15);color:#50c8b8}"
    ".art-entities{font-size:.7rem;color:#555;margin-top:1px}"
    ".error{color:#f87171;text-align:center;padding:20px}"
    ".empty{color:#666;text-align:center;padding:40px}"
    "</style></head><body>"
    "<div class='wrap'>"
    "<h1>Earnings Call Search</h1>"
    "<p class='sub'>Semantic search over earnings call transcripts — hear the exact moment.</p>"
    "<form class='search-row' id='form'>"
    "<input type='text' id='q' name='q' "
    "placeholder='e.g. Which CEOs mentioned tariffs in Q1 2025?' "
    "value='QUERY_PLACEHOLDER' autocomplete='off' autofocus>"
    "<button type='submit' id='btn'>Search</button>"
    "</form>"
    "<div class='spinner' id='spinner'>Searching &amp; fetching audio clips&#8230;</div>"
    "<div class='results' id='results'>RESULTS_PLACEHOLDER</div>"
    "</div>"
    "<script>"
    "const form=document.getElementById('form'),"
    "qInput=document.getElementById('q'),"
    "btn=document.getElementById('btn'),"
    "spin=document.getElementById('spinner'),"
    "res=document.getElementById('results');"
    "form.addEventListener('submit',async e=>{"
    "e.preventDefault();"
    "const q=qInput.value.trim();if(!q)return;"
    "btn.disabled=true;spin.style.display='block';res.innerHTML='';"
    "try{"
    "const r=await fetch('/search?q='+encodeURIComponent(q));"
    "res.innerHTML=await r.text();"
    "}catch(err){res.innerHTML='<p class=error>'+err+'</p>';}"
    "finally{btn.disabled=false;spin.style.display='none';}"
    "});"
    "document.addEventListener('click',function(e){"
    "const a=e.target.closest('a');"
    "const article=e.target.closest('.article');"
    "if(!article)return;"
    "if(a)return;"
    "const summary=article.querySelector('.article-summary');"
    "if(!summary)return;"
    "const expanded=article.classList.toggle('expanded');"
    "summary.style.display=expanded?'block':'-webkit-box';"
    "summary.style.webkitLineClamp=expanded?'unset':'2';"
    "summary.style.overflow=expanded?'visible':'hidden';"
    "summary.style.webkitBoxOrient=expanded?'unset':'vertical';"
    "});"
    "</script></body></html>"
)

RAINBOW = [
    "#7a3030", "#7a5530", "#7a7030", "#4a7a30",
    "#307a60", "#305a7a", "#553080"
]

def _rainbow_color(index: int) -> str:
    # Mirror without repeating endpoints: R O Y G B I V I B G Y O R O Y...
    forward = RAINBOW
    backward = RAINBOW[-2:0:-1]  # skip first and last to avoid repeats
    cycle = forward + backward
    return cycle[index % len(cycle)]


def _news_block(point_id: str) -> str:
    news = get_news_context(point_id)
    if "error" in news:
        return ""
    articles = news.get("articles", [])
    if not articles:
        return ""
    window = news.get("window", "")

    rows = []
    for i, a in enumerate(articles):
        title = a.get("title", "").replace("<", "&lt;").replace(">", "&gt;")
        url = a.get("url", "")
        src = a.get("source", "")
        pub = str(a.get("published_at", ""))[:10]
        lang = a.get("language", "")
        summ = a.get("summary", "").replace("<", "&lt;").replace(">", "&gt;")
        sentiment_val = a.get("sentiment")
        sentiment = sentiment_val if sentiment_val is not None else ""
        bias = a.get("bias", "")
        content_type = a.get("content_type", "")
        image_url = a.get("image_url", "")
        entities = a.get("entities") or {}
        persons = entities.get("Person", [])
        organizations = entities.get("Organization", [])
        locations = entities.get("Location", [])
        entity_str = " · ".join(str(e) for e in (persons + organizations + locations)[:5])

        title_html = (
            f'<a href="{url}" target="_blank" rel="noopener" '
            f'onclick="event.stopPropagation()">{title}</a>'
            if url else title
        )if url else title

        img_section = (
    f'<div class="article-image-wrap" style="'
    f'background:linear-gradient(135deg,{_rainbow_color(i)}cc 0%,{_rainbow_color(i)}44 100%);'
    f'flex-shrink:0;display:flex;align-items:center;justify-content:center;'
    f'border-right:1px solid {_rainbow_color(i)}66"></div>'
    )

        def sentiment_class(s):
            if s == 1:  return "art-badge art-badge-pos"
            if s == -1: return "art-badge art-badge-neg"
            return "art-badge art-badge-neu"

        def sentiment_label(s):
            if s == 1:  return "&#x2197; positive"
            if s == -1: return "&#x2198; negative"
            return "neutral"

        badges_html = ""
        badge_parts = []
        if sentiment != "":
            badge_parts.append(f'<span class="{sentiment_class(sentiment)}">Sentiment: {sentiment_label(sentiment)}</span>')
        if bias and bias != "None":
            badge_parts.append(f'<span class="art-badge art-badge-bias">Bias: {bias}</span>')
        if content_type and content_type != "article":
            badge_parts.append(f'<span class="art-badge art-badge-type">{content_type}</span>')
        if badge_parts:
            badges_html = f'<div class="art-badges">{"".join(badge_parts)}</div>'

        entity_html = f'<div class="art-entities">Mentions: {entity_str}</div>' if entity_str else ""

        rows.append(
            f'<div class="article">'
            f'{img_section}'
            f'<div class="article-body">'
            f'<div class="article-title">{title_html}</div>'
            f'<div class="article-meta">'
            f'<span class="meta-source">{src}</span>'
            + (f'<span class="meta-dot">·</span><span class="meta-date">{pub}</span>' if pub else "")
            + (f'<span class="meta-lang">{lang}</span>' if lang else "")
            + f'</div>'
            + (f'<div class="article-summary">{summ}</div>' if summ else "")
            + badges_html
            + entity_html
            + f'</div></div>'
        )

    window_span = f'<span class="news-window">{window}</span>' if window else ""
    return (
        f'<div class="news-section">'
        f'<div class="news-label">AskNews context {window_span}</div>'
        f'<div class="news-scroll">'
        + "\n".join(rows)
        + '</div></div>'
    )


def _ticker_class(ticker: str) -> str:
    return ticker.lower()


def _audio_block(point_id: str) -> str:
    clip = get_audio_clip(point_id)
    if "error" in clip:
        return f'<p class="no-clip">⚠ {clip["error"]}</p>'
    b64 = clip["audio_base64"]
    t0, t1 = clip.get("start_time", 0), clip.get("end_time", 0)
    label = f"{t0:.1f}s – {t1:.1f}s"
    return (
        f'<div class="audio-label">{label}</div>'
        f'<audio controls preload="auto">'
        f'<source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">'
        f'</audio>'
    )


def _render_results(query: str, ticker: str | None = None) -> str:
    results = search_earnings(query=query, ticker=ticker)
    if not results or "error" in results[0]:
        msg = results[0].get("error", "No results") if results else "No results"
        return f'<p class="empty">{msg}</p>'

    # Every result needs an audio clip and a news block, and each fetch is an
    # independent Qdrant round-trip. Submitting them all up front lets the
    # audio and news requests overlap, turning ~2N serial network calls into
    # one concurrent batch (~5x faster than fetching them one card at a time).
    point_ids = [r["point_id"] for r in results]
    with ThreadPoolExecutor(max_workers=min(16, 2 * len(point_ids))) as pool:
        audio_futures = {p: pool.submit(_audio_block, p) for p in point_ids}
        news_futures = {p: pool.submit(_news_block, p) for p in point_ids}
        audio_blocks = {p: f.result() for p, f in audio_futures.items()}
        news_blocks = {p: f.result() for p, f in news_futures.items()}

    cards = []
    for r in results:
        pid    = r["point_id"]
        audio  = audio_blocks[pid]
        news   = news_blocks[pid]
        ticker = r.get("ticker", "?")
        card = (
            f'<div class="card">'
            f'<div class="card-header">'
            f'<span class="badge {_ticker_class(ticker)}">{ticker}</span>'
            f'<span>{r.get("company","")}</span>'
            f'<span class="meta">{r.get("quarter","")} {r.get("year","")} &middot; {r.get("date","")}</span>'
            f'<span class="score">score {r.get("score",0):.3f}</span>'
            f'</div>'
            f'<div class="quote">{r.get("chunk_text","").replace("  "," ")}</div>'
            f'{audio}'
            f'{news}'
            f'</div>'
        )
        cards.append(card)
    return "\n".join(cards)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE.replace("QUERY_PLACEHOLDER", "").replace("RESULTS_PLACEHOLDER", "")


@app.get("/search", response_class=HTMLResponse)
def search(q: str = "", ticker: str = ""):
    if not q:
        return '<p class="empty">Enter a query above.</p>'
    html = _render_results(q, ticker=ticker or None)
    return html


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
