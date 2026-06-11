"""
Earnings Call Audio Search — web demo.

Run with:  .venv/bin/python app.py
Then open: http://localhost:8000
"""

import base64
import html as html_mod
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from mcp_server.server import (
    clone_collection,
    discover_earnings_calls,
    get_audio_clip,
    get_clone_status,
    get_indexing_jobs,
    get_news_context,
    get_news_graph,
    index_earnings_call,
    list_tickers,
    search_earnings,
)

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
    ".graph-section{background:#1a1d27;border:1px solid #2a2d3d;border-radius:12px;"
    "padding:16px 20px}"
    ".graph-out{margin-top:10px;display:flex;flex-direction:column;gap:10px}"
    ".graph-link{color:#8b9cf8;font-size:.85rem;font-weight:600;text-decoration:none;"
    "align-self:flex-start}"
    ".graph-link:hover{text-decoration:underline;color:#a8b8ff}"
    ".graph-chips{display:flex;gap:6px;flex-wrap:wrap}"
    ".graph-chip{display:inline-flex;align-items:center;gap:5px;background:#13151f;"
    "border:1px solid #2a2d3d;border-radius:20px;padding:3px 10px;font-size:.78rem;color:#ccc}"
    ".graph-chip .cnt{color:#666;font-size:.7rem}"
    ".graph-edges{display:flex;flex-direction:column;gap:4px}"
    ".graph-edge{font-size:.78rem;color:#999;line-height:1.5}"
    ".graph-edge .lbl{color:#50c8b8}"
    ".graph-query{color:#666;font-size:.74rem;font-style:italic;line-height:1.4}"
    ".graph-err{color:#f87171;font-size:.8rem}"
    ".error{color:#f87171;text-align:center;padding:20px}"
    ".empty{color:#666;text-align:center;padding:40px}"
    ".panel{background:#1a1d27;border:1px solid #2a2d3d;border-radius:12px;"
    "padding:14px 16px;margin-bottom:24px}"
    ".panel-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap}"
    ".panel-title{font-size:.78rem;font-weight:700;letter-spacing:.08em;"
    "text-transform:uppercase;color:#888}"
    ".panel-state{font-size:.78rem;color:#666;margin-right:auto}"
    ".panel-state.ok{color:#6dca6d}"
    ".btn-sm{padding:6px 14px;font-size:.82rem;border-radius:6px}"
    ".btn-ghost{background:transparent;border:1px solid #3a3d4d;color:#aaa}"
    ".btn-ghost:hover{background:#222533;border-color:#5b6af0}"
    ".chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}"
    ".chip{display:flex;align-items:center;gap:6px;background:#13151f;"
    "border:1px solid #2a2d3d;border-radius:20px;padding:4px 12px;font-size:.82rem}"
    ".chip .cnt{color:#666;font-size:.75rem}"
    ".chip.queued{border-style:dashed;color:#999}"
    ".ticker-form{display:none;margin-top:14px;border-top:1px solid #2a2d3d;"
    "padding-top:14px;grid-template-columns:repeat(3,1fr);gap:8px}"
    ".ticker-form.open{display:grid}"
    ".ticker-form input{padding:8px 10px;border-radius:6px;border:1px solid #333;"
    "background:#13151f;color:#fff;font-size:.85rem;outline:none}"
    ".ticker-form input:focus{border-color:#5b6af0}"
    ".ticker-form .full{grid-column:1/-1}"
    ".ingest-msg{font-size:.8rem;color:#888;margin-top:10px;white-space:pre-wrap}"
    ".ingest-msg.err{color:#f87171}"
    ".ingest-msg.ok{color:#6dca6d}"
    ".jobs{display:flex;flex-direction:column;gap:8px;margin-top:12px}"
    ".job{background:#13151f;border:1px solid #2a2d3d;border-radius:8px;padding:8px 12px}"
    ".job-head{display:flex;align-items:center;gap:8px;font-size:.82rem}"
    ".job-stage{color:#8b9cf8;font-size:.75rem}"
    ".job-pct{margin-left:auto;color:#aaa;font-size:.75rem}"
    ".job-bar{height:4px;background:#2a2d3d;border-radius:2px;margin-top:6px;overflow:hidden}"
    ".job-bar i{display:block;height:100%;background:#5b6af0;border-radius:2px;transition:width .5s}"
    ".job.done .job-bar i{background:#4ab44a}"
    ".job.failed .job-bar i{background:#dc3c3c}"
    ".job-err{color:#f87171;font-size:.75rem;margin-top:4px}"
    ".cands{display:flex;flex-direction:column;gap:6px;margin-top:10px}"
    ".cand{background:#13151f;border:1px solid #2a2d3d;border-radius:8px;padding:7px 12px;"
    "font-size:.82rem;cursor:pointer;display:flex;gap:8px;align-items:center}"
    ".cand:hover{border-color:#5b6af0}"
    ".cand .meta{margin-left:auto}"
    ".cand b{color:#8b9cf8;white-space:nowrap}"
    ".cand.off{opacity:.45;cursor:default}"
    ".cand.off:hover{border-color:#2a2d3d}"
    ".cand b{color:#8b9cf8;white-space:nowrap}"
    ".cand.off{opacity:.45;cursor:default}"
    ".cand.off:hover{border-color:#2a2d3d}"
    ".ai-summary{background:linear-gradient(135deg,#1a1d33,#1d1a2e);border:1px solid #3d3a6d;"
    "border-radius:12px;padding:18px 20px;margin-bottom:4px}"
    ".ai-summary-label{font-size:.75rem;font-weight:700;letter-spacing:.08em;color:#8b9cf8;"
    "text-transform:uppercase;margin-bottom:8px;display:flex;align-items:center;gap:6px}"
    ".ai-summary-model{font-size:.7rem;color:#555;font-weight:400;text-transform:none;"
    "letter-spacing:0;margin-left:auto}"
    ".ai-summary-text{font-size:.95rem;line-height:1.7;color:#d5d8ee;white-space:pre-wrap}"
    ".ai-summary-text q{color:#fff;font-style:italic}"
    ".cite{color:#8b9cf8;text-decoration:none;font-size:.78rem;font-weight:700;"
    "vertical-align:super;padding:0 1px}"
    ".cite:hover{color:#c3ccff;text-decoration:underline}"
    "html{scroll-behavior:smooth}"
    ".card:target{border-color:#8b9cf8;box-shadow:0 0 0 1px #8b9cf8}"
    ".result-num{font-size:.8rem;font-weight:700;color:#8b9cf8}"
    "</style></head><body>"
    "<div class='wrap'>"
    "<h1>Earnings Call Search</h1>"
    "<p class='sub'>Semantic search over earnings call transcripts — hear the exact moment.</p>"
    "<div class='panel' id='ingest-panel'>"
    "<div class='panel-head'>"
    "<span class='panel-title'>Indexed data</span>"
    "<span class='panel-state' id='clone-state'>checking…</span>"
    "<button class='btn-sm' id='clone-btn' style='display:none'>Clone to local</button>"
    "<button class='btn-sm btn-ghost' id='add-toggle' style='display:none'>+ Add ticker</button>"
    "</div>"
    "<div class='chips' id='chips'></div>"
    "<form class='ticker-form' id='add-form' autocomplete='off'>"
    "<input class='full' name='ticker' id='ticker-input' "
    "placeholder='Just a ticker, e.g. MSFT — calls are discovered for you' required>"
    "<button type='submit' class='btn-sm full' id='discover-btn'>Discover earnings calls</button>"
    "</form>"
    "<div class='cands' id='cands'></div>"
    "<div class='jobs' id='jobs'></div>"
    "<button class='btn-sm btn-ghost' id='clear-jobs' "
    "style='display:none;margin-top:8px'>Clear finished</button>"
    "<div class='ingest-msg' id='ingest-msg'></div>"
    "</div>"
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
    "document.addEventListener('click',async function(e){"
    "const b=e.target.closest('.graph-btn');"
    "if(!b)return;"
    "const sec=b.closest('.graph-section');"
    "const out=sec.querySelector('.graph-out');"
    "b.disabled=true;b.textContent='Building graph\\u2026';out.innerHTML='';"
    "try{"
    "const ctxEl=sec.querySelector('.graph-context');"
    "const context=ctxEl?ctxEl.textContent:'';"
    "const r=await(await fetch('/graph',{method:'POST',"
    "headers:{'Content-Type':'application/json'},"
    "body:JSON.stringify({q:sec.dataset.q||'',ticker:sec.dataset.ticker||'',context:context})})).json();"
    "if(r.error){"
    "out.innerHTML=`<div class='graph-err'>${esc(r.error)}</div>`;"
    "b.disabled=false;b.textContent='Build knowledge graph';return;}"
    "let h='';"
    "if(r.graph_query&&r.graph_query!==(sec.dataset.q||'')){"
    "h+=`<div class='graph-query'>graph query: ${esc(r.graph_query)}</div>`;}"
    "if(r.visualize_url){"
    "h+=`<a class='graph-link' href='${esc(r.visualize_url)}' target='_blank' rel='noopener'>"
    "Open interactive graph &#8599;</a>`;}"
    "const nodes=(r.nodes||[]).slice(0,12),edges=(r.edges||[]).slice(0,8);"
    "if(nodes.length){"
    "h+=`<div class='graph-chips'>`+nodes.map(n=>"
    "`<span class='graph-chip' title='${esc(n.type||'')}'>${esc(n.id)}"
    "<span class='cnt'>${n.count??''}</span></span>`).join('')+`</div>`;}"
    "if(edges.length){"
    "h+=`<div class='graph-edges'>`+edges.map(ed=>"
    "`<div class='graph-edge'>${esc(ed.from)} <span class='lbl'>&#8212;${esc(ed.label)}&#8594;</span> "
    "${esc(ed.to)}</div>`).join('')+`</div>`;}"
    "if(!h)h=`<div class='graph-err'>No graph data found for this call window.</div>`;"
    "out.innerHTML=h;b.style.display='none';"
    "}catch(err){"
    "out.innerHTML=`<div class='graph-err'>${esc(String(err))}</div>`;"
    "b.disabled=false;b.textContent='Build knowledge graph';}"
    "});"
    "const chips=document.getElementById('chips'),"
    "cloneState=document.getElementById('clone-state'),"
    "cloneBtn=document.getElementById('clone-btn'),"
    "addToggle=document.getElementById('add-toggle'),"
    "addForm=document.getElementById('add-form'),"
    "tickerInput=document.getElementById('ticker-input'),"
    "discoverBtn=document.getElementById('discover-btn'),"
    "cands=document.getElementById('cands'),"
    "jobsEl=document.getElementById('jobs'),"
    "clearBtn=document.getElementById('clear-jobs'),"
    "ingestMsg=document.getElementById('ingest-msg');"
    "let candList=[],candTicker='',pollTimer=null,prevActive=new Set();"
    "function msg(t,cls){ingestMsg.textContent=t;ingestMsg.className='ingest-msg '+(cls||'');}"
    "function esc(t){return String(t??'').replace(/</g,'&lt;').replace(/>/g,'&gt;');}"
    "async function loadStatus(){"
    "try{"
    "const s=await(await fetch('/ingest/status')).json();"
    "chips.innerHTML='';"
    "(s.tickers||[]).forEach(c=>{"
    "chips.insertAdjacentHTML('beforeend',"
    "`<span class='chip'><span class='badge ${c.ticker.toLowerCase()}'>${c.ticker}</span>"
    "${c.company||''} ${c.quarter||''} ${c.year||''} <span class='cnt'>${c.chunks} chunks</span></span>`);});"
    "(s.jobs||[]).forEach(j=>{"
    "chips.insertAdjacentHTML('beforeend',"
    "`<span class='chip queued'>${j.ticker} ${j.quarter} ${j.year} — ${j.stage||j.status}</span>`);});"
    "if(s.cloned){"
    "cloneState.textContent='local collection ✓ '+s.total_chunks+' chunks';"
    "cloneState.classList.add('ok');"
    "cloneBtn.style.display='none';addToggle.style.display='';"
    "}else{"
    "cloneState.textContent='no local collection — clone before adding tickers';"
    "cloneState.classList.remove('ok');"
    "cloneBtn.style.display='';addToggle.style.display='none';"
    "addForm.classList.remove('open');}"
    "}catch(e){cloneState.textContent='status unavailable';}}"
    "cloneBtn.addEventListener('click',async()=>{"
    "cloneBtn.disabled=true;cloneBtn.textContent='Cloning…';msg('Copying collection from the workshop cluster…');"
    "try{"
    "const r=await(await fetch('/ingest/clone',{method:'POST'})).json();"
    "if(r.error){msg(r.error,'err');}else{msg('Cloned '+r.local_points+' points to local.','ok');}"
    "}catch(e){msg(String(e),'err');}"
    "cloneBtn.disabled=false;cloneBtn.textContent='Clone to local';loadStatus();});"
    "addToggle.addEventListener('click',()=>addForm.classList.toggle('open'));"
    "function jobHtml(j){"
    "const cls=j.status==='done'?' done':j.status==='failed'?' failed':'';"
    "const pct=Math.round(j.percent||0);"
    "const label=j.status==='running'?(j.stage||'starting'):j.status;"
    "const err=j.error?`<div class='job-err'>${esc(j.error)}</div>`:'';"
    "return `<div class='job${cls}'><div class='job-head'>"
    "<b>${j.ticker} ${j.quarter} ${j.year}</b>"
    "<span class='job-stage'>${label}</span>"
    "<span class='job-pct'>${pct}%</span></div>"
    "<div class='job-bar'><i style='width:${pct}%'></i></div>${err}</div>`;}"
    "function startPolling(){if(!pollTimer)pollTimer=setInterval(loadJobs,2500);}"
    "function stopPolling(){if(pollTimer){clearInterval(pollTimer);pollTimer=null;}}"
    "async function loadJobs(){"
    "try{"
    "const list=await(await fetch('/ingest/jobs')).json();"
    "if(!Array.isArray(list))return;"
    "jobsEl.innerHTML=list.slice(0,8).map(jobHtml).join('');"
    "clearBtn.style.display=list.some(j=>['failed','cancelled','done'].includes(j.status))?'':'none';"
    "const active=new Set(list.filter(j=>j.status==='pending'||j.status==='running').map(j=>j.job_id));"
    "let finished=false;"
    "prevActive.forEach(id=>{if(!active.has(id))finished=true;});"
    "prevActive=active;"
    "if(finished)loadStatus();"
    "if(active.size){startPolling();}else{stopPolling();}"
    "}catch(e){}}"
    "function renderCands(list){"
    "candList=list;"
    "cands.innerHTML=list.map((c,i)=>{"
    "const off=c.already_indexed||!c.youtube_url||!c.quarter||!c.year;"
    "const conf=c.confidence!=null?Math.round(c.confidence*100)+'% conf':'';"
    "const note=c.already_indexed?'already indexed':conf;"
    "return `<div class='cand${off?' off':''}' data-i='${i}'>"
    "<b>${c.quarter||'?'} ${c.year||'?'}</b>"
    "<span>${esc(c.title||'').slice(0,80)}</span>"
    "<span class='meta'>${c.date||''}${note?' · '+note:''}</span></div>`;}).join('');}"
    "addForm.addEventListener('submit',async e=>{"
    "e.preventDefault();"
    "const ticker=tickerInput.value.trim().toUpperCase();"
    "if(!ticker)return;"
    "discoverBtn.disabled=true;discoverBtn.textContent='Discovering…';"
    "cands.innerHTML='';"
    "msg('Searching the web + news coverage for '+ticker+' earnings calls (takes ~10s)…');"
    "try{"
    "const r=await(await fetch('/ingest/discover?ticker='+encodeURIComponent(ticker))).json();"
    "if(r.error){msg(r.error,'err');}"
    "else{candTicker=r.ticker||ticker;renderCands(r.candidates||[]);"
    "msg((r.candidates||[]).length?'Pick a call below to start indexing.'"
    ":'No earnings-call recordings found for '+ticker+'.');}"
    "}catch(e){msg(String(e),'err');}"
    "discoverBtn.disabled=false;discoverBtn.textContent='Discover earnings calls';});"
    "cands.addEventListener('click',async e=>{"
    "const el=e.target.closest('.cand');"
    "if(!el||el.classList.contains('off'))return;"
    "const c=candList[+el.dataset.i];"
    "if(!c)return;"
    "msg('Starting indexing job for '+candTicker+' '+c.quarter+' '+c.year+'…');"
    "try{"
    "const r=await(await fetch('/ingest/add',{method:'POST',"
    "headers:{'Content-Type':'application/json'},"
    "body:JSON.stringify({youtube_url:c.youtube_url,ticker:candTicker,company:c.company,"
    "quarter:c.quarter,year:c.year,date:c.date})})).json();"
    "if(r.error){msg(r.error,'err');}"
    "else if(r.status==='already_indexed'){msg(r.detail,'err');el.classList.add('off');}"
    "else{msg('Indexing job started for '+candTicker+' '+c.quarter+' '+c.year+'.','ok');"
    "el.classList.add('off');await loadJobs();startPolling();loadStatus();}"
    "}catch(e){msg(String(e),'err');}});"
    "clearBtn.addEventListener('click',async()=>{"
    "clearBtn.disabled=true;"
    "try{await fetch('/ingest/jobs/clear',{method:'POST',"
    "headers:{'Content-Type':'application/json'},"
    "body:JSON.stringify({statuses:['failed','cancelled','done']})});}"
    "catch(e){}"
    "clearBtn.disabled=false;loadJobs();});"
    "loadStatus();loadJobs();"
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


SUMMARY_MODEL = "models/gemini-3.1-flash-lite"


def _ai_summary(query: str, results: list[dict]) -> tuple[str, str]:
    """Synthesize the retrieved chunks into a cited answer with Gemini Flash Lite.

    Sources are numbered 1..N in result order; bracketed citations in the model
    output become anchor links to the matching result card below.

    Returns a (html, plain_text) pair: the rendered HTML block, plus the raw
    plain-text summary used as HyDE grounding context for the knowledge graph.
    Both are "" when no summary could be produced.
    """
    import html as html_mod
    import os
    import re

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return "", ""
    try:
        from google import genai

        chunks = "\n\n".join(
            f"Source [{i}] — {r.get('company')} ({r.get('ticker')}) "
            f"{r.get('quarter')} {r.get('year')}, {r.get('speaker')}:\n"
            f"{(r.get('chunk_text') or '').strip()}"
            for i, r in enumerate(results, start=1)
        )
        prompt = (
            "You are summarizing earnings-call search results for an investor.\n"
            f'The user asked: "{query}"\n\n'
            "Below are numbered source chunks. Answer the question in 2-4 sentences "
            "using ONLY this material, naming companies and speakers where relevant. "
            "Then add up to 3 short bullet takeaways.\n"
            "Rules:\n"
            "- Cite sources inline with single bracketed numbers, e.g. [1] or [2][4]. "
            "Never combine numbers inside one bracket.\n"
            "- Where impactful, include a short verbatim quote (under 15 words) from a "
            "source, wrapped in double quotes, immediately followed by its citation.\n"
            "- Every sentence and bullet must carry at least one citation.\n"
            "- Plain text only, no markdown headers.\n\n"
            f"{chunks}"
        )
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=SUMMARY_MODEL, contents=prompt)
        text = (response.text or "").strip()
        if not text:
            return "", ""

        n = len(results)
        escaped = html_mod.escape(text)

        # Style "quoted" passages first (while the text is still plain),
        # then linkify citations — otherwise the regex would eat the
        # quotes inside the generated <a> attributes.
        styled = re.sub(r"&quot;(.+?)&quot;", r"<q>\1</q>", escaped)

        def _link(match: re.Match) -> str:
            idx = int(match.group(1))
            if 1 <= idx <= n:
                return f'<a class="cite" href="#result-{idx}">[{idx}]</a>'
            return match.group(0)

        linked = re.sub(r"\[(\d+)\]", _link, styled)

        html_block = (
            '<div class="ai-summary">'
            '<div class="ai-summary-label">AI summary'
            f'<span class="ai-summary-model">{SUMMARY_MODEL.split("/")[-1]}</span></div>'
            f'<div class="ai-summary-text">{linked}</div>'
            "</div>"
        )
        return html_block, text
    except Exception:
        return "", ""  # summary is best-effort; never block results


def _render_results(query: str, ticker: str | None = None) -> str:
    results = search_earnings(query=query, ticker=ticker)
    if not results or "error" in results[0]:
        msg = results[0].get("error", "No results") if results else "No results"
        return f'<p class="empty">{msg}</p>'

    # Fetch audio and news for all cards in parallel — news involves a live
    # AskNews call per point on a cold cache, so doing it sequentially
    # multiplies the latency by the number of results.
    from concurrent.futures import ThreadPoolExecutor

    pids = [r["point_id"] for r in results]
    with ThreadPoolExecutor(max_workers=10) as pool:
        summary_future = pool.submit(_ai_summary, query, results)
        audio_blocks = pool.map(_audio_block, pids)
        news_blocks = pool.map(_news_block, pids)
        audio_by_pid = dict(zip(pids, audio_blocks))
        news_by_pid = dict(zip(pids, news_blocks))
        summary_html, summary_text = summary_future.result()

    # One knowledge-graph block per search, right below the AI summary. The
    # plain summary text rides along in a hidden element so the lazy graph
    # builder can POST it as HyDE grounding context (escaped for the HTML
    # body; the browser decodes it back to plain text via textContent).
    graph_section = (
        f'<div class="graph-section" data-q="{html_mod.escape(query, quote=True)}" '
        f'data-ticker="{html_mod.escape(ticker or "", quote=True)}">'
        '<div class="news-label">Knowledge graph</div>'
        f'<div class="graph-context" hidden>{html_mod.escape(summary_text)}</div>'
        '<button type="button" class="btn-sm btn-ghost graph-btn">Build knowledge graph</button>'
        '<div class="graph-out"></div>'
        '</div>'
    )

    cards = [summary_html] if summary_html else []
    cards.append(graph_section)
    for i, r in enumerate(results, start=1):
        pid    = r["point_id"]
        audio  = audio_by_pid[pid]
        news   = news_by_pid[pid]
        ticker = r.get("ticker", "?")
        card = (
            f'<div class="card" id="result-{i}">'
            f'<div class="card-header">'
            f'<span class="result-num">[{i}]</span>'
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


# ── Ingestion panel API ───────────────────────────────────────────────────────

_clone_lock = threading.Lock()


def _local_collection_ready() -> bool:
    status = get_clone_status()
    return bool(status.get("exists")) and (status.get("points") or 0) > 0


@app.get("/ingest/status")
def ingest_status():
    status = get_clone_status()
    cloned = bool(status.get("exists")) and (status.get("points") or 0) > 0
    tickers: list = []
    jobs: list = []
    total = 0
    if cloned:
        snapshot = list_tickers()
        if "error" not in snapshot:
            tickers = snapshot["indexed"]
            jobs = snapshot["jobs"]
            total = snapshot["total_chunks"]
    return {
        "cloned": cloned,
        "tickers": tickers,
        "jobs": jobs,
        "total_chunks": total,
        "in_sync": status.get("in_sync"),
        "cloud_points": status.get("cloud_points"),
        "error": status.get("error"),
    }


@app.post("/ingest/clone")
def ingest_clone():
    if not _clone_lock.acquire(blocking=False):
        return {"error": "A clone is already in progress."}
    try:
        return clone_collection(force=True)
    finally:
        _clone_lock.release()


@app.get("/ingest/jobs")
def ingest_jobs():
    return get_indexing_jobs()


class ClearJobsRequest(BaseModel):
    statuses: list[str] = ["failed", "cancelled"]


@app.post("/ingest/jobs/clear")
def ingest_jobs_clear(req: ClearJobsRequest = ClearJobsRequest()):
    jobs_dir = Path(__file__).parent / "data" / "jobs"
    wanted = set(req.statuses) & {"failed", "cancelled", "done"}
    cleared = 0
    for path in jobs_dir.glob("*.json") if jobs_dir.exists() else []:
        try:
            if json.loads(path.read_text()).get("status") in wanted:
                path.unlink()
                cleared += 1
        except (json.JSONDecodeError, OSError):
            continue
    return {"cleared": cleared}


@app.get("/ingest/discover")
def ingest_discover(ticker: str, company: str = ""):
    if not ticker.strip():
        return {"error": "ticker is required"}
    return discover_earnings_calls(ticker=ticker, company=company)


class AddTickerRequest(BaseModel):
    youtube_url: str
    ticker: str
    quarter: str
    year: int
    company: str | None = None
    date: str | None = None


@app.post("/ingest/add")
def ingest_add(req: AddTickerRequest):
    # Adding new tickers is only supported once the collection is cloned
    # locally — the shared workshop cluster is read-only.
    if not _local_collection_ready():
        return {"error": "Clone the collection to local first — the workshop cluster is read-only."}
    return index_earnings_call(
        youtube_url=req.youtube_url,
        ticker=req.ticker,
        company=req.company or req.ticker,
        quarter=req.quarter,
        year=req.year,
        date=req.date or f"{req.year}-01-01",
    )


class GraphRequest(BaseModel):
    q: str
    ticker: str | None = None
    context: str | None = None


@app.post("/graph")
def graph(req: GraphRequest):
    if not req.q.strip():
        return {"error": "q is required"}
    return get_news_graph(
        req.q,
        ticker=(req.ticker or None),
        context=(req.context or None),
    )


@app.get("/search", response_class=HTMLResponse)
def search(q: str = "", ticker: str = ""):
    if not q:
        return '<p class="empty">Enter a query above.</p>'
    html = _render_results(q, ticker=ticker or None)
    return html


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
