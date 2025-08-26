# main.py
from typing import Dict, List, Optional, Tuple
import os, json, time, logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ==== RAG & (opsional) scraper ====
from rag_engine import query_index
try:
    from scraper import get_scraped_context  # optional
except Exception:
    get_scraped_context = None

# ==== OpenAI SDK baru ====
from openai import OpenAI

# ========================
# Build / Version metadata
# ========================
APP_VERSION = os.getenv("APP_VERSION", "dev")
COMMIT_SHA = os.getenv("COMMIT_SHA", "")
BUILD_TIME_ISO = os.getenv("BUILD_TIME", datetime.now(timezone.utc).isoformat())

# ========================
# App & CORS
# ========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_origins=["*", "http://localhost:3000"],  # TODO prod: batasi ke domain kamu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Logging & Rate limiting
# ========================
logging.basicConfig(level=logging.INFO, format="%(message)s")
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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _ST_MODEL

def _intent_bank_vectors():
    global _INTENT_BANK_VECS
    if not _INTENT_BANK_VECS:
        model = _lazy_load_st()
        _INTENT_BANK_VECS = model.encode(_INTENT_BANK, convert_to_numpy=True).tolist()
    return _INTENT_BANK_VECS

def _cos(a: List[float], b: List[float]) -> float:
    import math
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(y*y for y in b))
    return (num / (da*db)) if da and db else 0.0

def _semantic_score(text: str) -> float:
    if not text:
        return 0.0
    try:
        model = _lazy_load_st()
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

def _build_context(message: str) -> str:
    """Ambil konteks dari RAG; fallback ke scraper bila ada."""
    ctx = ""
    try:
        hits = query_index(message, top_k=4)  # [(score, text)]
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
    return ctx

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
    return f"""{style}
{scope}
{soft_gate}

CONTEXT:
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
# Chat endpoints
# ========================
@app.post("/chat")
async def chat(req: ChatRequest):
    if not _allow_request("chat", 20, 60):
        raise HTTPException(status_code=429, detail="Too Many Requests")

    lang = (req.lang or "de").lower()
    kw = _kw_match(req.message)
    sem = _semantic_score(req.message)
    intent_ok = bool(kw or sem >= 0.62)
    log_intent_analytics(req.message, kw, sem, "chat")

    ctx = _build_context(req.message)
    prompt = _build_prompt(req.message, ctx, lang, intent_ok)

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        reply_text = (resp.choices[0].message.content or "").strip()
        if not reply_text:
            reply_text = "Dazu habe ich keine gesicherte Information. Mehr hier: https://planville.de/kontakt" \
                         if lang == "de" else "I don't have verified info on that. More here: https://planville.de/kontakt"
        return {"reply": reply_text}
    except Exception:
        logging.exception("chat error")
        msg = "Ups, da ist etwas schiefgelaufen. Kontakt: https://planville.de/kontakt" if lang == "de" else \
              "Oops, something went wrong. Contact: https://planville.de/kontakt"
        return {"reply": msg}

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not _allow_request("chat_stream", 60, 60):
        raise HTTPException(status_code=429, detail="Too Many Requests")

    lang = (req.lang or "de").lower()
    kw = _kw_match(req.message)
    sem = _semantic_score(req.message)
    intent_ok = bool(kw or sem >= 0.62)
    log_intent_analytics(req.message, kw, sem, "chat_stream")

    ctx = _build_context(req.message)
    prompt = _build_prompt(req.message, ctx, lang, intent_ok)

    def token_stream():
        try:
            stream = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.get("content")
                except Exception:
                    delta = None
                if delta:
                    yield delta
        except Exception:
            msg = "Ups, da ist etwas schiefgelaufen. Bitte versuchen Sie es erneut. Kontakt: https://planville.de/kontakt" \
                  if lang == "de" else \
                  "Oops, something went wrong. Please try again. Contact: https://planville.de/kontakt"
            yield msg

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8", headers=headers)

@app.get("/chat/sse")
async def chat_sse(message: str = Query(...), lang: str = Query("de")):
    """Server-Sent Events: GET /chat/sse?message=...&lang=de"""
    lang = (lang or "de").lower()
    kw = _kw_match(message)
    sem = _semantic_score(message)
    intent_ok = bool(kw or sem >= 0.62)
    log_intent_analytics(message, kw, sem, "chat_sse")

    ctx = _build_context(message)
    prompt = _build_prompt(message, ctx, lang, intent_ok)

    def event_stream():
        try:
            stream = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.get("content")
                except Exception:
                    delta = None
                if delta:
                    yield "data: " + delta.replace("\n", "\\n") + "\\n\\n"
            yield "event: done\\ndata: [DONE]\\n\\n"
        except Exception:
            msg = "Ups, da ist etwas schiefgelaufen. Bitte versuchen Sie es erneut. Kontakt: https://planville.de/kontakt" \
                  if lang == "de" else \
                  "Oops, something went wrong. Please try again. Contact: https://planville.de/kontakt"
            yield "data: " + msg + "\\n\\n"

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_stream(), media_type="text/event-stream; charset=utf-8", headers=headers)

# ========================
# EKSEKUSI v2 Endpoints
# ========================
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "DRYRUN").upper()
CRM_API_URL = os.getenv("CRM_API_URL", "")
CRM_API_KEY = os.getenv("CRM_API_KEY", "")

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

@app.post("/funnel/next")
async def funnel_next(req: FunnelRequest):
    product = (req.product or "").lower()
    answered = req.answers_so_far or {}
    steps = {
        "pv": ["immobilientyp","eigentumer","bewohner","plz","dachform","dachflache_m2","ausrichtung","neigung_deg","verschattung","verbrauch_kwh","batterie","zeitrahmen"],
        "dach": ["eigentumer","dachform","material","baujahr","zustand","flaeche","neigung","daemmung","plz","zeitrahmen"],
        "wp":  ["eigentumer","bewohner","gebaeudetyp","baujahr","wohnflaeche","heizung","isolierung","aussenbereich","plz","zeitrahmen"],
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
        hits = query_index(req.message, top_k=3)  # [(score, text)]
        sources = [{"score": float(s), "text": str(t)[:280]} for (s, t) in hits]
        ctx = "\n\n".join([f"- {s['text']}" for s in sources])
        sys = ("Du bist der Planville Chat-Assistent. Antworte kurz und präzise nur aus dem gegebenen Kontext. "
               "Wenn Information nicht gesichert ist, sage: 'Dazu habe ich keine gesicherte Information. Ich kann dich gerne mit unserem Team verbinden.'")
        prompt = f"Kontext:\n{ctx}\n\nFrage: {req.message}\nAntwort ({req.lang}):"
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            temperature=0.0,
        )
        answer = (resp.choices[0].message.content or "").strip()
        if not answer:
            answer = sources[0]["text"] if sources else "Dazu habe ich keine gesicherte Information. Ich kann dich gerne mit unserem Team verbinden."
        return {"answer": answer, "sources": sources, "confidence": (sources[0]["score"] if sources else 0.0)}
    except Exception:
        logging.exception("ai_answer error")
        return {"answer": "Dazu habe ich keine gesicherte Information. Ich kann dich gerne mit unserem Team verbinden.", "sources": [], "confidence": 0.0}

@app.post("/lead")
async def push_lead(payload: LeadPayload):
    if EXECUTION_MODE != "LIVE" or not CRM_API_URL:
        logging.warning("[LEAD] DRY or CRM not configured — storing as accepted")
        return {"status": "accepted", "mode": EXECUTION_MODE, "crm": "not_configured"}
    try:
        import requests
        headers = {"Authorization": f"Bearer {CRM_API_KEY}", "Content-Type": "application/json"}
        r = requests.post(CRM_API_URL, headers=headers, json=payload.dict())
        return {"status": "ok", "crm_status": r.status_code, "crm_body": r.text}
    except Exception as e:
        logging.exception("CRM push failed")
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
    now = datetime.now(timezone.utc)
    slots: List[str] = []
    for d in range(1, 8):
        for hour in (10, 14, 17):
            dt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            dt = dt.replace(day=min(28, now.day) + d)
            slots.append(dt.isoformat())
    return {"plz": plz, "slots": slots[:12]}
