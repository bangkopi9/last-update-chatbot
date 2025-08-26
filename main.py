from typing import Dict, List
import json
from datetime import datetime, timezone
import time
import logging
from fastapi import Request, FastAPI
# --- Soft Intent Gate Utilities ---
VALID_KEYWORDS = [
    # German
    "photovoltaik","pv","solaranlage","dach","wärmepumpe","klimaanlage",
    "angebot","kosten","preise","förderung","termin","beratung",
    "installation","montage","wartung","service","garantie",
    # English
    "photovoltaics","solar","roof","heat pump","air conditioner","ac",
    "quote","cost","price","subsidy","appointment","consultation",
    "install","maintenance","warranty"
]

def _match_intent(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in VALID_KEYWORDS)

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import query_index
from scraper import get_scraped_context
from openai import OpenAI
import os
# --- Build/Version metadata ---
APP_VERSION = os.getenv("APP_VERSION", "dev")
COMMIT_SHA = os.getenv("COMMIT_SHA", "")
BUILD_TIME_ISO = os.getenv("BUILD_TIME", datetime.now(timezone.utc).isoformat())

# --- Logger ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
intent_logger = logging.getLogger("intent")

from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Inisialisasi FastAPI App
app = FastAPI()

# ✅ CORS Middleware – izinkan akses frontend lokal/frontend live
app.add_middleware(CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_origins=["*", "http://localhost:3000"],  # ⚠️ Production: ubah ke ["https://planville.de"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Inisialisasi OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Struktur Permintaan dari Frontend
class ChatRequest(BaseModel):

def _build_prompt(user_message: str, context_text: str, lang: str, intent_ok: bool) -> str:
    cta = "Weitere Fragen? Kontakt: https://planville.de/kontakt" if lang == "de" else "More questions? Contact: https://planville.de/kontakt"
    style = "Antworte präzise, professionell und freundlich." if lang == "de" else "Answer concisely, professionally, and helpfully."
    scope = (
        "Thema: Photovoltaik, Dachsanierung, Wärmepumpe, Klimaanlage. "
        "Antworte auf Basis des CONTEXT unten. Wenn CONTEXT nicht ausreicht, antworte kurz (1–2 Sätze) und füge am Ende den CTA hinzu."
        if lang == "de"
        else "Topics: Photovoltaics, roofing, heat pumps, air conditioning. "
             "Answer based on CONTEXT below. If CONTEXT is insufficient, reply briefly (1–2 sentences) and append the CTA."
    )
    soft_gate = (
        "Falls die Nutzerfrage klar außerhalb der Themen ist, antworte sehr kurz (1–2 Sätze) und füge den CTA hinzu."
        if lang == "de"
        else "If the user question is clearly off-topic, answer very briefly (1–2 sentences) and append the CTA."
    )

    prompt = f"""{style}
    {scope}
    {soft_gate}

    CONTEXT:
    {context_text}

    USER:
    {user_message}

    ASSISTANT (CTA am Ende falls nötig / append CTA if needed):
    """
    return prompt

    message: str
    lang: str = "de"  # default bahasa Jerman

# ✅ Keyword-based intent detection (untuk keamanan jawaban)
VALID_KEYWORDS = [
    "photovoltaik", "photovoltaics", "dach", "roof",
    "wärmepumpe", "heat pump", "klimaanlage", "air conditioner",
    "beratung", "consultation", "angebot", "quote",
    "kontakt", "contact", "termin", "appointment", "montage", "installation"
]

def is_valid_intent(message: str) -> bool:
    """Periksa apakah input user mengandung keyword valid"""
    msg = message.lower()
    return any(keyword in msg for keyword in VALID_KEYWORDS)

# ✅ Endpoint utama chatbot
@app.post("/chat")
async def chat(request: Request, request: ChatRequest):
    # Rate limit
    if not _allow_request('chat', 20, 60):
        from fastapi import HTTPException
        raise HTTPException(status_code=429, detail='Too Many Requests')
    print(f"[📨 Request] Language: {request.lang} | Message: {request.message}")
    # --- Soft intent gate ---
    intent_kw = _match_intent(request.message if not False else message)
sem_score = _semantic_score(request.message if not False else message)
intent_sem = bool(sem_score >= 0.62)
intent_ok = bool(intent_kw or intent_sem)
log_intent_analytics((request.message if not False else message), intent_kw, sem_score, 'chat')
    # Build context from RAG / scraper if available
    try:
        context_text = ''
        if 'query_index' in globals():
            try:
                ctx = query_index(request.message, k=4)
                if isinstance(ctx, list):
                    context_text = '\n'.join(ctx)
                else:
                    context_text = str(ctx)
            except Exception:
                pass
        # Fallback scraper if available
        if not context_text and 'get_scraped_context' in globals():
            try:
                sc = get_scraped_context(request.message)
                if sc:
                    context_text = sc
            except Exception:
                pass
    except Exception:
        context_text = ''

    prompt = _build_prompt(request.message, context_text, request.lang or 'de', intent_ok)

    # 🔒 Filter input: hanya pertanyaan yang sesuai keyword
    if not is_valid_intent(request.message):
        fallback_msg = {
            "de": "Ich kann nur Fragen zu Planville Dienstleistungen beantworten. "
                  "Bitte kontaktieren Sie uns direkt unter: https://planville.de/kontakt",
            "en": "I can only answer questions related to Planville services. "
                  "Please contact us directly here: https://planville.de/kontakt"
        }
        return {"reply": fallback_msg.get(request.lang, fallback_msg["de"])}

    try:
        # 🧠 Ambil konteks dari RAG index
        context_docs = query_index(request.message)

        # 🔄 Jika tidak ada hasil RAG, fallback ke hasil scraping
        if not context_docs:
            print("[⚠️] RAG kosong → menggunakan fallback scraper.")
            context_docs = get_scraped_context(request.message)

        # 🔗 Gabungkan semua dokumen hasil jadi konteks
        context_text = "\n".join(context_docs)

        # 📝 Bangun prompt untuk GPT
        prompt = f"""
Du bist ein professioneller Kundenservice-Assistent von Planville GmbH.
Antworte bitte höflich, direkt und hilfreich basierend auf dem folgenden Kontext.

🔎 Frage:
{request.message}

📄 Kontext:
{context_text}
"""

        # 🤖 Kirim ke OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        # ✅ Ambil jawaban
        reply_text = response.choices[0].message.content.strip()

        # 🔁 Jika kosong, fallback ke jawaban statis
        if not reply_text:
            fallback = (
                "Entschuldigung, ich habe leider keine passende Information zu Ihrer Anfrage.\n\n"
                "📞 Kontaktieren Sie unser Team direkt:\n"
                "👉 https://planville.de/kontakt"
            )
            return {"reply": fallback}

        return {"reply": reply_text}

    except Exception as e:
        print(f"[❌ GPT ERROR]: {e}")
        return {
            "reply": (
                "Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut "
                "oder kontaktieren Sie uns direkt.\n\n➡️ https://planville.de/kontakt"
            )
        }

# ✅ Optional: Endpoint healthcheck
@app.get("/healthz")
def health_check():
    return {"status": "ok"}


from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream(request: Request, request: ChatRequest):
    # Rate limit
    if not _allow_request('chat_stream', 60, 60):
        from fastapi import HTTPException
        raise HTTPException(status_code=429, detail='Too Many Requests')
    """
    Streaming chunked response (text/plain). Frontend reads the stream and appends tokens live.
    """
    lang = request.lang or "de"
    intent_kw = _match_intent(request.message if not False else message)
sem_score = _semantic_score(request.message if not False else message)
intent_sem = bool(sem_score >= 0.62)
intent_ok = bool(intent_kw or intent_sem)
log_intent_analytics((request.message if not False else message), intent_kw, sem_score, 'chat_stream')

    # Build context (same as /chat)
    context_text = ''
    try:
        if 'query_index' in globals():
            try:
                ctx = query_index(request.message, k=4)
                if isinstance(ctx, list):
                    context_text = '\n'.join(ctx)
                else:
                    context_text = str(ctx)
            except Exception:
                pass
        if not context_text and 'get_scraped_context' in globals():
            try:
                sc = get_scraped_context(request.message)
                if sc:
                    context_text = sc
            except Exception:
                pass
    except Exception:
        context_text = ''

    prompt = _build_prompt(request.message, context_text, lang, intent_ok)

    def token_stream():
        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.get("content")
                except Exception:
                    delta = None
                if delta:
                    yield delta
        except Exception as e:
            msg = "Ups, da ist etwas schiefgelaufen. Bitte versuchen Sie es erneut. Kontakt: https://planville.de/kontakt" if lang=="de" else \
                  "Oops, something went wrong. Please try again. Contact: https://planville.de/kontakt"
            yield msg

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8", headers=headers)



from fastapi import Request, Query
@app.get("/chat/sse")
async def chat_sse(message: str = Query(...), lang: str = Query("de")):
    """
    Server-Sent Events (text/event-stream) endpoint.
    Usage: GET /chat/sse?message=...&lang=de
    """
    lang = lang or "de"
    intent_kw = _match_intent(request.message if not True else message)
sem_score = _semantic_score(request.message if not True else message)
intent_sem = bool(sem_score >= 0.62)
intent_ok = bool(intent_kw or intent_sem)
log_intent_analytics((request.message if not True else message), intent_kw, sem_score, 'chat_sse')

    # Build context (same as others)
    context_text = ''
    try:
        if 'query_index' in globals():
            try:
                ctx = query_index(message, k=4)
                if isinstance(ctx, list):
                    context_text = '\n'.join(ctx)
                else:
                    context_text = str(ctx)
            except Exception:
                pass
        if not context_text and 'get_scraped_context' in globals():
            try:
                sc = get_scraped_context(message)
                if sc:
                    context_text = sc
            except Exception:
                pass
    except Exception:
        context_text = ''

    prompt = _build_prompt(message, context_text, lang, intent_ok)

    def event_stream():
        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.get("content")
                except Exception:
                    delta = None
                if delta:
                    yield "data: " + delta.replace("\n", "\\n") + "\\n\\n"
            # Signal end
            yield "event: done\\ndata: [DONE]\\n\\n"
        except Exception:
            msg = "Ups, da ist etwas schiefgelaufen. Bitte versuchen Sie es erneut. Kontakt: https://planville.de/kontakt" if lang=="de" else \
                  "Oops, something went wrong. Please try again. Contact: https://planville.de/kontakt"
            yield "data: " + msg + "\\n\\n"

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_stream(), media_type="text/event-stream; charset=utf-8", headers=headers)

# --- Simple in-memory rate limiter ---
from collections import defaultdict, deque
REQUEST_BUCKETS: Dict[str, deque] = defaultdict(deque)

def _allow_request(bucket: str, limit: int, window_sec: int) -> bool:
    now = time.time()
    q = REQUEST_BUCKETS[bucket]
    # purge
    while q and (now - q[0]) > window_sec:
        q.popleft()
    if len(q) >= limit:
        return False
    q.append(now)
    return True

# --- Intent analytics ---
INTENT_LOG_PATH = os.getenv("INTENT_LOG_PATH")

def log_intent_analytics(text: str, kw_hit: bool, sem_score: float, source: str):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "kw": bool(kw_hit),
        "sem_score": float(sem_score or 0.0),
        "text": text[:512]
    }
    try:
        intent_logger.info(json.dumps(rec, ensure_ascii=False))
    except Exception:
        pass
    if INTENT_LOG_PATH:
        try:
            with open(INTENT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

def _semantic_score(text: str) -> float:
    if not text:
        return 0.0
    v = _embed_with_st(text)
    if v is None:
        v = _embed_with_openai(text)
    if v is None:
        return 0.0
    bank = _intent_bank_vectors()
    if not bank:
        return 0.0
    mx = 0.0
    for b in bank:
        try:
            num = sum(x*y for x,y in zip(v,b))
            da = sum(x*x for x in v) ** 0.5
            db = sum(y*y for y in b) ** 0.5
            sc = (num / (da*db)) if da and db else 0.0
        except Exception:
            sc = 0.0
        if sc > mx:
            mx = sc
    return float(mx)

@app.get("/version")
async def version():
    return {
        "version": APP_VERSION,
        "commit": COMMIT_SHA,
        "build_time": BUILD_TIME_ISO
    }


# ========================
# 🔌 Added Endpoints (EKSEKUSI v2)
# ========================
import os
from fastapi import Body
from pydantic import BaseModel, Field
from typing import Optional

EXECUTION_MODE = os.getenv("EXECUTION_MODE", "LIVE").upper()
CRM_API_URL = os.getenv("CRM_API_URL", "")
CRM_API_KEY = os.getenv("CRM_API_KEY", "")

class FunnelRequest(BaseModel):
    product: str
    answers_so_far: Dict = {}
    session_id: str | None = None

class AIRequest(BaseModel):
    message: str
    lang: str = "de"
    context_pack_ids: List[str] | None = None

class LeadPayload(BaseModel):
    source: str = "chatbot"
    product: str
    qualification: Dict = {}
    contact: Dict = {}
    score: int = 0
    disqualified: bool = False
    notes: str | None = None
    meta: Dict = {}

@app.post("/funnel/next")
async def funnel_next(req: FunnelRequest):
    # Minimal example logic; expand with real rules
    product = (req.product or "").lower()
    answered = req.answers_so_far or {}
    steps = {
        "pv": ["immobilientyp","eigentumer","bewohner","plz","dachform","dachflache_m2","ausrichtung","neigung_deg","verschattung","verbrauch_kwh","batterie","zeitrahmen"],
        "dach": ["eigentumer","dachform","material","baujahr","zustand","flaeche","neigung","daemmung","plz","zeitrahmen"],
        "wp": ["eigentumer","bewohner","gebaeudetyp","baujahr","wohnflaeche","heizung","isolierung","aussenbereich","plz","zeitrahmen"],
        "mieterstrom": ["objekttyp","einheiten","zaehler","eigentumer","plz"]
    }
    key = "pv" if "pv" in product else ("dach" if "dach" in product else ("wp" if "wärme" in product or "wp"==product else ("mieterstrom" if "mieter" in product else "pv")))
    flow = steps.get(key, steps["pv"])
    next_slots = [s for s in flow if s not in answered]
    disqualified = False
    if key in ("pv","dach","wp"):
        if answered.get("eigentumer") in (False, "nein", "Nein", "no", "No"):
            disqualified = True
    percent = int(100 * (len(flow)-len(next_slots)) / max(1, len(flow)))
    return {"product": key, "next_slot": (next_slots[0] if not disqualified and next_slots else None), "percent": percent, "disqualified": disqualified}

@app.post("/ai/answer")
async def ai_answer(req: AIRequest):
    # Try RAG
    try:
        from rag_engine import query_index
        hits = query_index(req.message, top_k=3)
        sources = [{"score": float(s), "text": t[:280]} for (s,t) in hits]
        # Try OpenAI if available
        try:
            import openai
            client = openai.OpenAI()
            sys = "Du bist der Planville Chat-Assistent. Antworte kurz und präzise nur aus dem gegebenen Kontext. Wenn Information nicht gesichert ist, sage: 'Dazu habe ich keine gesicherte Information. Ich kann dich gerne mit unserem Team verbinden.'"
            ctx = "\n\n".join([f"- {s['text']}" for s in sources])
            prompt = f"Kontext:\n{ctx}\n\nFrage: {req.message}\nAntwort ({req.lang}):"
            resp = client.chat.completions.create(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"), messages=[{"role":"system","content":sys},{"role":"user","content":prompt}], temperature=0)
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            # Fallback template
            answer = (sources[0]["text"] if sources else "Dazu habe ich keine gesicherte Information. Ich kann dich gerne mit unserem Team verbinden.")
        return {"answer": answer, "sources": sources, "confidence": (sources[0]["score"] if sources else 0.0)}
    except Exception as e:
        return {"answer": "Dazu habe ich keine gesicherte Information. Ich kann dich gerne mit unserem Team verbinden.", "sources": [], "confidence": 0.0}

@app.post("/lead")
async def push_lead(payload: LeadPayload):
    # LIVE: push to CRM if configured
    if EXECUTION_MODE != "LIVE" or not CRM_API_URL:
        logging.warning("[LEAD] DRY or CRM not configured — storing as accepted")
        return {"status":"accepted","mode":EXECUTION_MODE,"crm":"not_configured"}
    try:
        import requests
        headers={"Authorization": f"Bearer {CRM_API_KEY}", "Content-Type":"application/json"}
        r = requests.post(CRM_API_URL, headers=headers, json=payload.dict())
        return {"status":"ok","crm_status": r.status_code, "crm_body": r.text}
    except Exception as e:
        logging.exception("CRM push failed")
        return {"status":"error","error": str(e)}

@app.post("/track")
async def track(event: Dict = Body(...)):
    try:
        event = event or {}
        event["ts"]= int(time.time()*1000)
        # In real-world, send to analytics sink
        print("[TRACK]", json.dumps(event)[:500])
    except Exception as e:
        logging.exception("track error")
    return {"ok": True}

@app.get("/schedule/suggest")
async def schedule_suggest(plz: str = ""):
    # Stubbed schedule slots (next 7 days, 10:00/14:00/17:00)
    now = datetime.now()
    slots = []
    for d in range(1,8):
        for hour in (10,14,17):
            dt = datetime(now.year, now.month, min(28, now.day)+d, hour, 0, tzinfo=timezone.utc)
            slots.append(dt.isoformat())
    return {"plz": plz, "slots": slots[:12]}
