"""
Spoken-claim fact-checker — record or upload a conversation, transcribe it,
pull out the factual claims, and verify each against the live web.

Pipeline:
  audio (upload or mic)
    -> transcribe            (Gemini gemini-2.5-flash)
    -> extract + fact-check  (Gemini + Google Search grounding)  ........ "fast pass"
    -> optional deep dive    (AskNews DeepNews: news/web/X/wiki)  ........ on demand, per claim

This is NOT limited to the earnings-call corpus — it checks the open world.
Verdicts are evidence-based (supported / contradicted / misleading /
unverifiable / incoherent), always with sources, never an oracle.

Run:   .venv/bin/python -m uvicorn factcheck_app:app --port 8030
Open:  http://localhost:8030
"""

import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from google import genai
from google.genai import types

GEMINI_MODEL = "gemini-2.5-flash"

app = FastAPI()
_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))

# Audio formats Gemini ingests directly; anything else (e.g. mic webm) is converted.
GEMINI_AUDIO_MIMES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/aiff",
    "audio/aac", "audio/ogg", "audio/flac", "audio/mp4", "audio/x-m4a",
}


# ── Pipeline ────────────────────────────────────────────────────────────────

def _to_supported_audio(data: bytes, mime: str) -> tuple[bytes, str]:
    """Return audio in a Gemini-supported format, converting via ffmpeg if needed."""
    if mime in GEMINI_AUDIO_MIMES:
        return data, mime
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        from pydub import AudioSegment

        seg = AudioSegment.from_file(io.BytesIO(data))
        out = io.BytesIO()
        seg.export(out, format="mp3")
        return out.getvalue(), "audio/mpeg"
    except Exception as exc:
        raise RuntimeError(f"could not convert audio ({mime}): {exc}")


def transcribe(data: bytes, mime: str) -> str:
    data, mime = _to_supported_audio(data, mime)
    resp = _genai.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=data, mime_type=mime),
            "Transcribe this audio verbatim. Label distinct speakers as "
            "'Speaker 1:', 'Speaker 2:', etc. Return only the transcript text.",
        ],
    )
    return (resp.text or "").strip()


_FACTCHECK_PROMPT = """You are a rigorous, skeptical fact-checker. Below is a transcript of a conversation.

1. Extract every distinct, checkable FACTUAL claim. Ignore opinions, greetings, and small talk.
2. For each claim, USE GOOGLE SEARCH to verify it against current, real-world sources.
3. Assign a verdict:
   - "supported"    : reliable sources clearly confirm it
   - "contradicted" : reliable sources clearly refute it
   - "misleading"   : partially true but distorted or missing key context
   - "unverifiable" : no reliable source settles it
   - "incoherent"   : the statement does not logically make sense
4. Write a one-sentence assessment, and include the source URLs you actually relied on.

Return ONLY a JSON array (no prose, no markdown fences):
[{"claim":"<the claim>","verdict":"supported","assessment":"<one sentence>","sources":["https://..."]}]

TRANSCRIPT:
\"\"\"
__TRANSCRIPT__
\"\"\""""


def _extract_json_array(text: str) -> list[dict]:
    s, e = text.find("["), text.rfind("]")
    if s != -1 and e != -1 and e > s:
        return json.loads(text[s : e + 1])
    raise ValueError("model did not return a JSON array")


def fact_check(transcript: str) -> list[dict]:
    resp = _genai.models.generate_content(
        model=GEMINI_MODEL,
        contents=[_FACTCHECK_PROMPT.replace("__TRANSCRIPT__", transcript)],
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0,
        ),
    )
    claims = _extract_json_array((resp.text or "").strip())

    # Fallback: if the model gave no per-claim URLs, offer the grounding sources.
    grounding: list[str] = []
    try:
        gm = resp.candidates[0].grounding_metadata
        for ch in (getattr(gm, "grounding_chunks", None) or []):
            w = getattr(ch, "web", None)
            if w and getattr(w, "uri", None):
                grounding.append(w.uri)
    except Exception:
        pass
    for c in claims:
        if not c.get("sources") and grounding:
            c["sources"] = grounding[:3]
    return claims


def deep_dive(claim: str) -> dict[str, Any]:
    """Deep multi-source research on a single claim via AskNews DeepNews."""
    from asknews_sdk import AskNewsSDK
    from asknews_sdk.dto.deepnews import (
        AnthropicTextDelta,
        ContentBlockDeltaEvent,
        CreateDeepNewsResponseStreamChunkV2,
        CreateDeepNewsResponseStreamSource,
        CreateDeepNewsResponseStreamSourcesNewsSource,
        CreateDeepNewsResponseStreamSourcesWebSource,
    )

    key = os.getenv("ASKNEWS_API_KEY", "")
    if not key:
        return {"error": "ASKNEWS_API_KEY not set — deep dive unavailable."}

    ask = AskNewsSDK(api_key=key)
    query = (
        f'Fact-check this claim against news, the web, Wikipedia and X: "{claim}". '
        "Search broadly, weigh the evidence, and conclude whether it is true, false, "
        "misleading, or unverifiable, with reasons."
    )
    resp = ask.chat.get_deep_news(
        messages=[{"role": "user", "content": query}],
        search_depth=1, max_depth=4, sources=["asknews", "google", "x", "wiki"],
        stream=True, return_sources=True, model="claude-sonnet-4-6",
        engine="v2.0", only_cited_sources=True,
    )

    parts: list[str] = []
    sources: list[dict] = []
    seen: set[str] = set()
    for m in resp:
        if isinstance(m, CreateDeepNewsResponseStreamChunkV2):
            ev = m.choices[0].delta
            if isinstance(ev, ContentBlockDeltaEvent) and isinstance(ev.delta, AnthropicTextDelta):
                parts.append(ev.delta.text)
            continue
        if not isinstance(m, CreateDeepNewsResponseStreamSource):
            continue
        src = m.source
        if isinstance(src, CreateDeepNewsResponseStreamSourcesNewsSource):
            it = src.data
            k = str(it.article_id)
            if k in seen:
                continue
            seen.add(k)
            sources.append({"title": it.eng_title or it.title, "url": str(it.article_url), "source": it.source_id})
        elif isinstance(src, CreateDeepNewsResponseStreamSourcesWebSource):
            it = src.data
            k = str(it.url)
            if k in seen:
                continue
            seen.add(k)
            sources.append({"title": it.title, "url": str(it.url), "source": it.source})

    full = "".join(parts)
    a, b = full.find("<final_answer>"), full.find("</final_answer>")
    analysis = full[a + len("<final_answer>") : b].strip() if a != -1 and b != -1 else full.strip()
    return {"analysis": analysis, "sources": sources}


# ── HTTP ──────────────────────────────────────────────────────────────────────

@app.post("/factcheck")
async def factcheck_endpoint(req: Request):
    body = await req.json()
    b64 = body.get("audio_b64", "")
    mime = body.get("mime", "audio/mpeg")
    if not b64:
        return JSONResponse({"error": "no audio provided"}, status_code=400)
    try:
        data = base64.b64decode(b64)
        transcript = transcribe(data, mime)
        if not transcript:
            return JSONResponse({"error": "transcription was empty"}, status_code=422)
        claims = fact_check(transcript)
        return {"transcript": transcript, "claims": claims}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/deepdive")
async def deepdive_endpoint(req: Request):
    body = await req.json()
    claim = (body.get("claim") or "").strip()
    if not claim:
        return JSONResponse({"error": "no claim provided"}, status_code=400)
    try:
        return deep_dive(claim)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def index():
    return _PAGE


_PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Conversation Fact-Checker</title><style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1117;color:#e0e0e0;min-height:100vh;padding:32px 16px}
.wrap{max-width:820px;margin:0 auto}
h1{font-size:1.6rem;color:#fff;margin-bottom:4px}
.sub{color:#888;font-size:.9rem;margin-bottom:24px}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:#1a1d27;border:1px solid #2a2d3d;border-radius:12px;padding:18px;margin-bottom:20px}
button{padding:11px 20px;border-radius:8px;border:none;background:#5b6af0;color:#fff;font-size:.95rem;font-weight:600;cursor:pointer}
button:hover{background:#6b7af8}button:disabled{opacity:.5;cursor:not-allowed}
button.rec{background:#cc3344}button.rec.on{background:#e0455a;animation:pulse 1.2s infinite}
@keyframes pulse{50%{opacity:.55}}
.or{color:#666;font-size:.85rem}
label.file{padding:11px 20px;border-radius:8px;border:1px solid #444;background:#13151f;color:#ccc;cursor:pointer;font-size:.95rem}
label.file:hover{border-color:#5b6af0}
#fname{color:#888;font-size:.82rem}
.status{color:#888;font-size:.88rem;margin:16px 0;display:none}
.transcript{background:#13151f;border:1px solid #2a2d3d;border-radius:10px;padding:14px 16px;margin-bottom:20px;font-size:.88rem;line-height:1.6;color:#bbb;white-space:pre-wrap;display:none}
.transcript b{color:#888;font-size:.74rem;text-transform:uppercase;letter-spacing:.08em;display:block;margin-bottom:8px}
.claim{background:#1a1d27;border:1px solid #2a2d3d;border-left:4px solid #555;border-radius:10px;padding:16px;margin-bottom:14px}
.claim .q{font-size:.98rem;color:#eee;line-height:1.5;margin-bottom:8px}
.verdict{display:inline-block;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;padding:3px 10px;border-radius:20px;margin-bottom:8px}
.v-supported{background:rgba(74,180,74,.16);color:#6dca6d}.bl-supported{border-left-color:#4ab44a}
.v-contradicted{background:rgba(220,60,60,.16);color:#e87070}.bl-contradicted{border-left-color:#dc3c3c}
.v-misleading{background:rgba(220,160,40,.16);color:#e0b050}.bl-misleading{border-left-color:#dca028}
.v-unverifiable{background:rgba(150,150,150,.16);color:#aaa}.bl-unverifiable{border-left-color:#777}
.v-incoherent{background:rgba(150,100,220,.16);color:#b69cf0}.bl-incoherent{border-left-color:#9664dc}
.assess{font-size:.88rem;color:#bbb;line-height:1.55;margin-bottom:8px}
.srcs{font-size:.78rem;color:#777}.srcs a{color:#8b9cf8;text-decoration:none;margin-right:10px}.srcs a:hover{text-decoration:underline}
.dd-btn{margin-top:10px;padding:6px 14px;font-size:.8rem;background:#2a2d3d}.dd-btn:hover{background:#3a3d4d}
.dd{margin-top:12px;padding:12px;background:#13151f;border:1px solid #2a2d3d;border-radius:8px;font-size:.85rem;line-height:1.6;color:#bbb;white-space:pre-wrap;display:none}
.err{color:#f87171;padding:16px}
</style></head><body><div class=wrap>
<h1>Conversation Fact-Checker</h1>
<p class=sub>Record or upload a conversation. It's transcribed, broken into factual claims, and each is checked against the live web — not just the earnings corpus.</p>
<div class=controls>
<button id=rec class=rec>● Record</button>
<span class=or>or</span>
<label class=file>Upload audio<input id=file type=file accept="audio/*" hidden></label>
<span id=fname></span>
<button id=go disabled>Fact-check</button>
</div>
<div class=status id=status></div>
<div class=transcript id=transcript></div>
<div id=results></div>
</div><script>
const recBtn=document.getElementById('rec'),fileIn=document.getElementById('file'),
fname=document.getElementById('fname'),go=document.getElementById('go'),
status=document.getElementById('status'),tEl=document.getElementById('transcript'),
res=document.getElementById('results');
let blob=null,mime=null,mr=null,recording=false,chunks=[];

function setBlob(b,m,label){blob=b;mime=m;go.disabled=false;fname.textContent=label;}

recBtn.onclick=async()=>{
 if(!recording){
  try{const s=await navigator.mediaDevices.getUserMedia({audio:true});
   mr=new MediaRecorder(s);chunks=[];
   mr.ondataavailable=e=>chunks.push(e.data);
   mr.onstop=()=>{const b=new Blob(chunks,{type:mr.mimeType});
    setBlob(b,(mr.mimeType||'audio/webm').split(';')[0],'🎙️ recording ('+Math.round(b.size/1024)+' KB)');
    s.getTracks().forEach(t=>t.stop());};
   mr.start();recording=true;recBtn.textContent='■ Stop';recBtn.classList.add('on');
  }catch(e){status.style.display='block';status.textContent='Mic error: '+e;}
 }else{mr.stop();recording=false;recBtn.textContent='● Record';recBtn.classList.remove('on');}
};
fileIn.onchange=()=>{const f=fileIn.files[0];if(f)setBlob(f,f.type||'audio/mpeg','📎 '+f.name);};

function b64(b){return new Promise(r=>{const fr=new FileReader();fr.onload=()=>r(fr.result.split(',')[1]);fr.readAsDataURL(b);});}

go.onclick=async()=>{
 if(!blob)return;
 go.disabled=true;res.innerHTML='';tEl.style.display='none';
 status.style.display='block';status.textContent='Transcribing & checking claims against the web…';
 try{
  const audio=await b64(blob);
  const r=await fetch('/factcheck',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({audio_b64:audio,mime:mime})});
  const d=await r.json();
  if(d.error){status.style.display='none';res.innerHTML='<p class=err>'+d.error+'</p>';go.disabled=false;return;}
  status.style.display='none';
  tEl.style.display='block';tEl.innerHTML='<b>Transcript</b>'+esc(d.transcript);
  res.innerHTML=(d.claims||[]).map(card).join('')||'<p class=err>No checkable claims found.</p>';
 }catch(e){status.style.display='none';res.innerHTML='<p class=err>'+e+'</p>';}
 go.disabled=false;
};
function esc(s){return (s||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function card(c,i){
 const v=(c.verdict||'unverifiable').toLowerCase();
 const srcs=(c.sources||[]).map(u=>'<a href="'+u+'" target=_blank rel=noopener>'+(new URL(u).hostname.replace('www.',''))+'</a>').join('');
 return '<div class="claim bl-'+v+'">'
  +'<span class="verdict v-'+v+'">'+v+'</span>'
  +'<div class=q>'+esc(c.claim)+'</div>'
  +'<div class=assess>'+esc(c.assessment||'')+'</div>'
  +(srcs?'<div class=srcs>'+srcs+'</div>':'')
  +'<button class="dd-btn" onclick="deepdive(this,'+i+')" data-claim="'+esc(c.claim).replace(/"/g,'&quot;')+'">🔎 Deep dive (AskNews)</button>'
  +'<div class=dd></div></div>';
}
async function deepdive(btn,i){
 const claim=btn.getAttribute('data-claim');const box=btn.nextElementSibling;
 btn.disabled=true;box.style.display='block';box.textContent='Researching across news, web, Wikipedia & X… (~1–2 min)';
 try{
  const r=await fetch('/deepdive',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({claim})});
  const d=await r.json();
  if(d.error){box.textContent='Error: '+d.error;btn.disabled=false;return;}
  const srcs=(d.sources||[]).slice(0,8).map(s=>'<a href="'+s.url+'" target=_blank rel=noopener>'+esc(s.source||s.title||'source')+'</a>').join(' · ');
  box.innerHTML=esc(d.analysis||'(no analysis)')+(srcs?'<div class=srcs style="margin-top:10px">'+srcs+'</div>':'');
 }catch(e){box.textContent='Error: '+e;}
 btn.disabled=false;
}
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8030, log_level="warning")
