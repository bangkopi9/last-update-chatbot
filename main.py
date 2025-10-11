# main.py — Planville/Wattson Backend (latency-optimized, p95 terasa ≤ 5 detik)
from typing import Dict, List, Optional, Tuple
import os, json, time, logging, asyncio
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException, Body, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ==== RAG & (opsional) scraper ====
from rag_engine import query_index
try:
    from scraper import get_scraped_context  # optional
except Exception:
    get_scraped_context = None

# ==== OpenAI SDK (tanpa custom httpx) ====
from openai import OpenAI

# ==== Database ====
try:
    from database import get_db, get_all_lead_types, create_leadchatbot, test_connection
    DATABASE_AVAILABLE = True
except Exception as e:
    DATABASE_AVAILABLE = False
    logging.warning(f"Database not available: {e}")

# ========================
# Build / Version metadata
# ========================
APP_VERSION    = os.getenv("APP_VERSION", "dev")
COMMIT_SHA     = os.getenv("COMMIT_SHA", "")
BUILD_TIME_ISO = os.getenv("BUILD_TIME", datetime.now(timezone.utc).isoformat())

# ========================
# Konfigurasi runtime (ENV)
# ========================
OPENAI_MODEL        = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT      = float(os.getenv("OPENAI_TIMEOUT", "30"))   # detik
OPENAI_TEMPERATURE  = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
# ↓↓↓ Batasi output supaya selesai cepat (target 3–5 dtk)
OPENAI_MAX_TOKENS   = int(os.getenv("OPENAI_MAX_TOKENS", "110"))

RAG_TOP_K           = int(os.getenv("RAG_TOP_K", "3"))
# ↓↓↓ Hard cap waktu RAG/scraper agar tidak menahan TTFB/stream
RAG_TIMEOUT         = float(os.getenv("RAG_TIMEOUT", "0.7"))     # detik

ST_DISABLE          = os.getenv("ST_DISABLE", "0") in ("1", "true", "True")
PREWARM_ST          = os.getenv("PREWARM_ST", "1") in ("1", "true", "True")

EXECUTION_MODE      = os.getenv("EXECUTION_MODE", "DRYRUN").upper()
CRM_API_URL         = os.getenv("CRM_API_URL", "")
CRM_API_KEY         = os.getenv("CRM_API_KEY", "")

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

# ========================
# App & CORS
# ========================
app = FastAPI()

_ALLOWED        = os.getenv("ALLOWED_ORIGINS", "").strip()
_ALLOWED_REGEX  = os.getenv("ALLOWED_ORIGIN_REGEX", r"^https://.*\.vercel\.app$").strip()

if _ALLOWED:
    _origins = [o.strip() for o in _ALLOWED.split(",") if o.strip()]
    _allow_credentials = os.getenv("ALLOW_CREDENTIALS", "false").lower() == "true"
else:
    _origins = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5500"]
    _allow_credentials = False

if any(o == "*" for o in _origins):
    _allow_credentials = False  # aman untuk wildcard

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=_ALLOWED_REGEX if _ALLOWED_REGEX else None,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ========================
# Logging & simple rate limit
# ========================
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("app")
intent_logger = logging.getLogger("intent")

from collections import defaultdict, deque
REQUEST_BUCKETS: Dict[str, deque] = defaultdict(deque)

def _allow_request(bucket: str, limit: int, window_sec: int) -> bool:
    now = time.time()
    q = REQUEST_BUCKETS[bucket]
    while q and (now - q[0]) > window_sec:
        q.popleft()
    if len(q) >= limit:
        return False
    q.append(now)
    return True

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        dt = (time.perf_counter() - t0) * 1000
        try:
            log.info(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "method": request.method,
                "path": request.url.path,
                "ms": round(dt, 1)
            }))
        except Exception:
            pass

def log_intent_analytics(text: str, kw_hit: bool, sem_score: float, source: str):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "kw": bool(kw_hit),
        "sem_score": float(sem_score or 0.0),
        "text": (text or "")[:512],
    }
    try:
        intent_logger.info(json.dumps(rec, ensure_ascii=False))
    except Exception:
        pass

# ========================
# OpenAI client
# ========================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=OPENAI_TIMEOUT,
)

# ========================
# Soft intent gate (keyword + semantic)
# ========================
VALID_KEYWORDS = [
    # de
    "photovoltaik","pv","solaranlage","dach","wärmepumpe","klimaanlage",
    "angebot","kosten","preise","förderung","termin","beratung",
    "installation","montage","wartung","service","garantie",
    # en
    "photovoltaics","solar","roof","heat pump","air conditioner","ac",
    "quote","cost","price","subsidy","appointment","consultation",
    "install","maintenance","warranty"
]

def _kw_match(text: str) -> bool:
    return bool(text) and any(k in text.lower() for k in VALID_KEYWORDS)

_ST_MODEL = None
_INTENT_BANK = [
    # de
    "Angebot Photovoltaik", "PV Anlage Dach", "Dachsanierung Kosten",
    "Wärmepumpe Beratung", "Klimaanlage Installation", "Termin Beratung",
    "Förderung Photovoltaik", "Montage PV", "Service Wartung PV",
    # en
    "photovoltaics quote", "solar panels on roof", "roof renovation",
    "heat pump consultation", "air conditioner install", "book appointment",
]
_INTENT_BANK_VECS: List[List[float]] = []

def _lazy_load_st():
    if ST_DISABLE:
        return None
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _ST_MODEL

def _intent_bank_vectors():
    if ST_DISABLE:
        return []
    global _INTENT_BANK_VECS
    if not _INTENT_BANK_VECS:
        model = _lazy_load_st()
        if model is None:
            return []
        _INTENT_BANK_VECS = model.encode(_INTENT_BANK, convert_to_numpy=True).tolist()
    return _INTENT_BANK_VECS

def _cos(a: List[float], b: List[float]) -> float:
    import math
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(y*y for y in b))
    return (num / (da*db)) if da and db else 0.0

def _semantic_score(text: str) -> float:
    if ST_DISABLE or not text:
        return 0.0
    try:
        model = _lazy_load_st()
        if model is None:
            return 0.0
        v = model.encode([text], convert_to_numpy=True)[0].tolist()
        bank = _intent_bank_vectors()
        return max((_cos(v, b) for b in bank), default=0.0)
    except Exception:
        return 0.0

# ========================
# Data models & prompt
# ========================
class ChatRequest(BaseModel):
    message: str
    lang: str = "de"

class FunnelRequest(BaseModel):
    product: str
    answers_so_far: Dict = {}
    session_id: Optional[str] = None

class AIRequest(BaseModel):
    message: str
    lang: str = "de"
    context_pack_ids: Optional[List[str]] = None

class LeadPayload(BaseModel):
    source: str = "chatbot"
    product: str
    qualification: Dict = {}
    contact: Dict = {}
    score: int = 0
    disqualified: bool = False
    notes: Optional[str] = None
    meta: Dict = {}

def _build_context_sync(message: str) -> str:
    """Ambil konteks dari RAG; fallback ke scraper bila ada. (SINKRON)"""
    ctx = ""
    try:
        hits = query_index(message, top_k=RAG_TOP_K)  # [(score, text)]
        if isinstance(hits, list) and hits:
            ctx = "\n".join([t for (_, t) in hits if isinstance(t, str)])
    except Exception:
        pass
    if not ctx and get_scraped_context:
        try:
            sc = get_scraped_context(message)
            if sc:
                ctx = sc
        except Exception:
            pass
    # potong keras untuk jaga total token input
    return (ctx or "")[:8000]

async def _build_context_timeboxed(message: str, timeout_s: float = None) -> str:
    """Jalankan _build_context_sync di threadpool + timeout ketat agar tidak menahan TTFB/stream."""
    timeout_s = RAG_TIMEOUT if timeout_s is None else timeout_s
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(loop.run_in_executor(None, _build_context_sync, message), timeout=timeout_s)
    except asyncio.TimeoutError:
        return ""

def _build_prompt(user_message: str, context_text: str, lang: str, intent_ok: bool) -> str:
    cta = "Weitere Fragen? Kontakt: https://planville.de/kontakt" if lang == "de" else \
          "More questions? Contact: https://planville.de/kontakt"
    style = "Antworte präzise, professionell und freundlich." if lang == "de" else \
            "Answer concisely, professionally, and helpfully."
    scope = ("Thema: Photovoltaik, Dachsanierung, Wärmepumpe, Klimaanlage. "
             "Antworte NUR auf Basis des CONTEXT unten. Wenn CONTEXT nicht ausreicht, "
             "antworte kurz (1–2 Sätze) und füge den CTA hinzu."
             if lang == "de" else
             "Topics: Photovoltaics, roofing, heat pumps, air conditioning. "
             "Answer ONLY based on the CONTEXT below. If CONTEXT is insufficient, "
             "reply briefly (1–2 sentences) and append the CTA.")
    soft_gate = ("Falls die Frage off-topic ist, antworte sehr kurz (1–2 Sätze) + CTA."
                 if lang == "de" else
                 "If the question is off-topic, answer very briefly (1–2 sentences) + CTA.")
    gate_hint = "" if intent_ok else ("[OFFTOPIC WARNING]\n")
    return f"""{style}
{scope}
{soft_gate}

{gate_hint}CONTEXT:
{context_text}

USER:
{user_message}

ASSISTANT (append CTA if needed):
"""

# ========================
# Health & Version
# ========================
@app.get("/version")
async def version():
    return {"version": APP_VERSION, "commit": COMMIT_SHA, "build_time": BUILD_TIME_ISO}

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

# ========================
# Prewarm on startup
# ========================
@app.on_event("startup")
def _prewarm():
    if not PREWARM_ST:
        return
    try:
        if not ST_DISABLE:
            _ = _lazy_load_st()
            if _ is not None:
                _.encode(["warmup"], convert_to_numpy=True)
        try:
            _ = query_index("warmup", top_k=1)
        except Exception:
            pass
        log.info(json.dumps({"event": "prewarm_done"}))
    except Exception as e:
        log.info(json.dumps({"event": "prewarm_skip", "err": str(e)}))

# ========================
# Util: panggil OpenAI (non-stream)
# ========================
def _openai_complete(prompt: str):
    t2 = time.perf_counter()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
    )
    openai_ms = (time.perf_counter() - t2) * 1000
    return resp, openai_ms

# ========================
# Chat (non-stream)
# ========================
@app.post("/chat")
async def chat(req: ChatRequest):
    if not _allow_request("chat", 20, 60):
        raise HTTPException(status_code=429, detail="Too Many Requests")

    t0 = time.perf_counter()
    lang = (req.lang or "de").lower()

    # Intent check dipersingkat agar tidak memblok
    kw = _kw_match(req.message)
    sem = 0.0
    try:
        sem = _semantic_score(req.message) if not ST_DISABLE else 0.0
    except Exception:
        sem = 0.0
    intent_ok = bool(kw or sem >= 0.62)
    log_intent_analytics(req.message, kw, sem, "chat")

    # RAG time-boxed
    t_ctx0 = time.perf_counter()
    ctx = await _build_context_timeboxed(req.message, RAG_TIMEOUT)
    ctx_ms = (time.perf_counter() - t_ctx0) * 1000

    t_p0 = time.perf_counter()
    prompt = _build_prompt(req.message, ctx, lang, intent_ok)
    prompt_ms = (time.perf_counter() - t_p0) * 1000

    try:
        resp, openai_ms = _openai_complete(prompt)
        reply_text = (resp.choices[0].message.content or "").strip()
        if not reply_text:
            reply_text = ("Dazu habe ich keine gesicherte Information. Mehr hier: https://planville.de/kontakt"
                          if lang == "de" else
                          "I don't have verified info on that. More here: https://planville.de/kontakt")

        total_ms = (time.perf_counter() - t0) * 1000
        log.info(json.dumps({
            "event": "chat_perf",
            "prep_ms": round(ctx_ms + prompt_ms, 1),
            "ctx_ms": round(ctx_ms, 1),
            "prompt_ms": round(prompt_ms, 1),
            "openai_ms": round(openai_ms, 1),
            "total_ms": round(total_ms, 1)
        }))
        return {"reply": reply_text}
    except Exception:
        logging.exception("chat error")
        msg = ("Ups, da ist etwas schiefgelaufen. Kontakt: https://planville.de/kontakt"
               if lang == "de" else
               "Oops, something went wrong. Contact: https://planville.de/kontakt")
        return {"reply": msg}

# ========================
# Chat (streaming NDJSON) — TTFB < 1s, RAG ≤ 0.7s, token stream cepat
# ========================
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not _allow_request("chat_stream", 60, 60):
        raise HTTPException(status_code=429, detail="Too Many Requests")

    lang = (req.lang or "de").lower()

    # >>> Jangan lakukan pekerjaan berat di luar generator,
    #     supaya kita bisa kirim "preface" dulu untuk TTFB cepat.

    async def generator():
        t0 = time.perf_counter()

        # 0) Preface seketika → FE sembunyikan skeleton (TTFB < 1 dtk)
        yield json.dumps({"type": "preface", "text": "Moment…"}) + "\n"

        # 1) Soft intent check (tidak menghambat; cepat saja)
        kw = _kw_match(req.message)
        try:
            sem = _semantic_score(req.message) if not ST_DISABLE else 0.0
        except Exception:
            sem = 0.0
        intent_ok = bool(kw or sem >= 0.62)
        log_intent_analytics(req.message, kw, sem, "chat_stream")

        # 2) RAG time-boxed (≤ RAG_TIMEOUT detik)
        t_ctx0 = time.perf_counter()
        ctx = await _build_context_timeboxed(req.message, RAG_TIMEOUT)
        ctx_ms = (time.perf_counter() - t_ctx0) * 1000

        # 3) Prompt siap (ringkas)
        t_p0 = time.perf_counter()
        prompt = _build_prompt(req.message, ctx, lang, intent_ok)
        prompt_ms = (time.perf_counter() - t_p0) * 1000

        # 4) Mulai stream ke OpenAI — kirim marker "start" lalu token demi token
        openai_ttfb_ms = None
        t_oo = time.perf_counter()
        try:
            stream = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=OPENAI_TEMPERATURE,
                max_tokens=OPENAI_MAX_TOKENS,
                stream=True,
            )
            first = True
            yield json.dumps({"type": "start"}) + "\n"
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.get("content")
                except Exception:
                    delta = None
                if delta:
                    if openai_ttfb_ms is None:
                        openai_ttfb_ms = (time.perf_counter() - t_oo) * 1000
                    # FE kamu mengharapkan "token" (bukan "chunk")
                    yield json.dumps({"type": "token", "text": delta}) + "\n"
        except Exception:
            msg = ("Ups, da ist etwas schiefgelaufen. Bitte versuchen Sie es erneut. Kontakt: https://planville.de/kontakt"
                   if lang == "de" else
                   "Oops, something went wrong. Please try again. Contact: https://planville.de/kontakt")
            yield json.dumps({"type": "error", "text": msg}) + "\n"
        finally:
            total_ms = (time.perf_counter() - t0) * 1000
            try:
                log.info(json.dumps({
                    "event": "chat_stream_perf",
                    "prep_ms": round(ctx_ms + prompt_ms, 1),
                    "ctx_ms": round(ctx_ms, 1),
                    "prompt_ms": round(prompt_ms, 1),
                    "openai_ttfb_ms": None if openai_ttfb_ms is None else round(openai_ttfb_ms, 1),
                    "total_ms": round(total_ms, 1)
                }))
            except Exception:
                pass
            yield json.dumps({"type": "metric", "ttlb": round(total_ms/1000.0, 3)}) + "\n"
            yield json.dumps({"type": "end"}) + "\n"

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "X-Accel-Buffering": "no",
        # berguna untuk debugging cepat di FE
        # "X-Perf-PrepMS": akan di-log via event metric
    }
    return StreamingResponse(generator(), media_type="application/x-ndjson; charset=utf-8", headers=headers)

# ========================
# SSE opsional (GET) — dibiarkan, tapi pola sama bisa diterapkan jika dipakai
# ========================
@app.get("/chat/sse")
async def chat_sse(message: str = Query(...), lang: str = Query("de")):
    lang = (lang or "de").lower()

    # (Catatan: endpoint ini jarang dipakai. Bila dipakai, idealnya juga kirim "ready" cepat
    #  lalu time-box RAG seperti di atas. Untuk singkatnya, kita biarkan mendekati aslinya.)

    kw = _kw_match(message)
    try:
        sem = _semantic_score(message) if not ST_DISABLE else 0.0
    except Exception:
        sem = 0.0
    intent_ok = bool(kw or sem >= 0.62)
    log_intent_analytics(message, kw, sem, "chat_sse")

    t_ctx0 = time.perf_counter()
    ctx = await _build_context_timeboxed(message, RAG_TIMEOUT)
    ctx_ms = (time.perf_counter() - t_ctx0) * 1000
    t_p0 = time.perf_counter()
    prompt = _build_prompt(message, ctx, lang, intent_ok)
    prompt_ms = (time.perf_counter() - t_p0) * 1000

    def event_stream():
        yield "event: ready\ndata: ok\n\n"
        t_oo = time.perf_counter()
        openai_ttfb_ms = None
        try:
            stream = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=OPENAI_TEMPERATURE,
                max_tokens=OPENAI_MAX_TOKENS,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.get("content")
                except Exception:
                    delta = None
                if delta:
                    if openai_ttfb_ms is None:
                        openai_ttfb_ms = (time.perf_counter() - t_oo) * 1000
                    safe = delta.replace("\n", "\\n")
                    yield f"data: {safe}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        except Exception:
            msg = ("Ups, da ist etwas schiefgelaufen. Bitte versuchen Sie es erneut. Kontakt: https://planville.de/kontakt"
                   if lang == "de" else
                   "Oops, something went wrong. Please try again. Contact: https://planville.de/kontakt")
            yield f"data: {msg}\n\n"
        finally:
            total_ms = (time.perf_counter() - t_p0) * 1000 + ctx_ms
            try:
                log.info(json.dumps({
                    "event": "chat_sse_perf",
                    "prep_ms": round(ctx_ms + prompt_ms, 1),
                    "ctx_ms": round(ctx_ms, 1),
                    "prompt_ms": round(prompt_ms, 1),
                    "openai_ttfb_ms": None if openai_ttfb_ms is None else round(openai_ttfb_ms, 1),
                    "total_ms": round(total_ms, 1)
                }))
            except Exception:
                pass

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "X-Accel-Buffering": "no",
        "X-Perf-PrepMS": str(round(ctx_ms + prompt_ms, 1)),
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream; charset=utf-8", headers=headers)

# ========================
# EKSEKUSI v2 Endpoints (dipertahankan)
# ========================
class FunnelNextResponse(BaseModel):
    product: str
    next_slot: Optional[str]
    percent: int
    disqualified: bool

@app.post("/funnel/next")
async def funnel_next(req: FunnelRequest) -> FunnelNextResponse:
    product = (req.product or "").lower()
    answered = req.answers_so_far or {}
    steps = {
        "pv":   ["immobilientyp","eigentumer","bewohner","plz","dachform","dachflache_m2","ausrichtung","neigung_deg","verschattung","verbrauch_kwh","batterie","zeitrahmen"],
        "dach": ["eigentumer","dachform","material","baujahr","zustand","flaeche","neigung","daemmung","plz","zeitrahmen"],
        "wp":   ["eigentumer","bewohner","gebaeudetyp","baujahr","wohnflaeche","heizung","isolierung","aussenbereich","plz","zeitrahmen"],
        "mieterstrom": ["objekttyp","einheiten","zaehler","eigentumer","plz"]
    }
    key = "pv" if "pv" in product else ("dach" if "dach" in product else ("wp" if ("wärme" in product or "wp"==product) else ("mieterstrom" if "mieter" in product else "pv")))
    flow = steps.get(key, steps["pv"])
    next_slots = [s for s in flow if s not in answered]
    disqualified = False
    if key in ("pv","dach","wp") and str(answered.get("eigentumer")).lower() in ("false","nein","no"):
        disqualified = True
    percent = int(100 * (len(flow) - len(next_slots)) / max(1, len(flow)))
    return {"product": key, "next_slot": (None if disqualified or not next_slots else next_slots[0]), "percent": percent, "disqualified": disqualified}

@app.post("/ai/answer")
async def ai_answer(req: AIRequest):
    try:
        hits = query_index(req.message, top_k=min(3, RAG_TOP_K))  # [(score, text)]
        sources = [{"score": float(s), "text": str(t)[:280]} for (s, t) in hits]
        ctx = "\n\n".join([f"- {s['text']}" for s in sources])
        sys = ("Du bist der Planville Chat-Assistent. Antworte kurz und präzise nur aus dem gegebenen Kontext. "
               "Wenn Information nicht gesichert ist, sage: 'Dazu habe ich keine gesicherte Information. Ich kann dich gerne mit unserem Team verbinden.'")
        prompt = f"Kontext:\n{ctx}\n\nFrage: {req.message}\nAntwort ({req.lang}):"
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=OPENAI_MAX_TOKENS,
        )
        openai_ms = (time.perf_counter() - t0) * 1000
        answer = (resp.choices[0].message.content or "").strip()
        if not answer:
            answer = sources[0]["text"] if sources else "Dazu habe ich keine gesicherte Information. Ich kann dich gerne mit unserem Team verbinden."
        log.info(json.dumps({"event": "ai_answer_perf", "openai_ms": round(openai_ms, 1)}))
        return {"answer": answer, "sources": sources, "confidence": (sources[0]["score"] if sources else 0.0)}
    except Exception:
        logging.exception("ai_answer error")
        return {"answer": "Dazu habe ich keine gesicherte Information. Ich kann dich gerne mit unserem Team verbinden.", "sources": [], "confidence": 0.0}

@app.get("/leadtype")
async def get_lead_types():
    """
    Get all available lead types from the database.
    This is used by the chatbot frontend to populate the lead type dropdown.
    """
    if not DATABASE_AVAILABLE:
        return {"error": "Database not available", "lead_types": []}

    try:
        db = next(get_db())
        try:
            lead_types = get_all_lead_types(db)
            return {
                "lead_types": [
                    {
                        "id": lt.id,
                        "key": lt.key,
                        "name": lt.name,
                    }
                    for lt in lead_types
                ]
            }
        finally:
            db.close()
    except Exception as e:
        logging.exception("Error fetching lead types")
        return {"error": str(e), "lead_types": []}


@app.get("/places/autocomplete")
async def places_autocomplete(input: str = Query(..., min_length=1)):
    """
    Google Places API autocomplete proxy.
    Returns address predictions for German addresses.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "https://maps.googleapis.com/maps/api/place/autocomplete/json",
                params={
                    "input": input,
                    "key": GOOGLE_PLACES_API_KEY,
                    "components": "country:de",
                    "types": "address",
                    "language": "de"
                }
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logging.exception("Google Places autocomplete error")
        return {"status": "error", "predictions": [], "error": str(e)}


@app.get("/places/details")
async def places_details(place_id: str = Query(...)):
    """
    Google Places API place details proxy.
    Returns detailed information about a specific place including geometry and address components.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "https://maps.googleapis.com/maps/api/place/details/json",
                params={
                    "place_id": place_id,
                    "key": GOOGLE_PLACES_API_KEY,
                    "fields": "formatted_address,address_components,geometry",
                    "language": "de"
                }
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logging.exception("Google Places details error")
        return {"status": "error", "result": {}, "error": str(e)}


class ChatbotLeadPayload(BaseModel):
    """Payload for chatbot lead submission"""
    # Contact info (required)
    gender: str
    first_name: str
    last_name: str
    email: str
    phone1: str

    # Contact info (optional)
    company: Optional[str] = None
    phone2: Optional[str] = None
    phone3: Optional[str] = None

    # Address (required)
    street_and_number: str
    zip_and_city: str
    province: str
    latitude: float
    longitude: float

    # Lead metadata (required)
    lead_type_id: int
    notes: str

    # Lead metadata (optional)
    session_id: Optional[str] = None


@app.post("/lead")
async def push_lead(payload: ChatbotLeadPayload):
    """
    Create a new lead from chatbot.
    Inserts into LeadChatbot table directly (no auth needed).
    A worker/cron will process it later into the Lead table.
    """
    if not DATABASE_AVAILABLE:
        logging.error("[LEAD] Database not available")
        return {"status": "error", "error": "Database not available"}

    try:
        db = next(get_db())
        try:
            # Create LeadChatbot entry with hardcoded source
            lead_chatbot = create_leadchatbot(
                db,
                gender=payload.gender,
                first_name=payload.first_name,
                last_name=payload.last_name,
                company=payload.company,
                street_and_number=payload.street_and_number,
                zip_and_city=payload.zip_and_city,
                province=payload.province,
                latitude=payload.latitude,
                longitude=payload.longitude,
                email=payload.email,
                phone1=payload.phone1,
                phone2=payload.phone2,
                phone3=payload.phone3,
                lead_type_id=payload.lead_type_id,
                source="Wattson/Chatbot",  # Hardcoded source
                session_id=payload.session_id,
                notes=payload.notes,
            )

            log.info(json.dumps({
                "event": "lead_created",
                "lead_chatbot_id": lead_chatbot.id,
                "email": payload.email,
                "lead_type_id": payload.lead_type_id,
            }))

            return {
                "status": "ok",
                "lead_chatbot_id": lead_chatbot.id,
                "message": "Lead stored successfully. Will be processed by cron job."
            }
        finally:
            db.close()

    except Exception as e:
        logging.exception("Error creating lead")
        return {"status": "error", "error": str(e)}

@app.post("/track")
async def track(event: Dict = Body(...)):
    try:
        event = event or {}
        event["ts"] = int(time.time() * 1000)
        print("[TRACK]", json.dumps(event)[:500])
    except Exception:
        logging.exception("track error")
    return {"ok": True}

@app.get("/schedule/suggest")
async def schedule_suggest(plz: str = ""):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    slots: List[str] = []
    for d in range(1, 8):
        for hour in (10, 14, 17):
            dt = (now + timedelta(days=d)).replace(hour=hour)
            slots.append(dt.isoformat())
    return {"plz": plz, "slots": slots[:12]}
