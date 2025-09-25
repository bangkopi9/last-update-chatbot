# rag_engine.py — fast, cached, score-aware (L2 → 0..1)
# Kompatibel dengan main.py (time-boxed dari threadpool)
from __future__ import annotations
import os, json, re
from typing import List, Tuple, Any, Optional

import numpy as np

# ==== FAISS opsional (fallback ke NumPy jika tidak ada) ====
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# ========================
# 📂 Paths & ENV
# ========================
DATA_DIR       = os.getenv("RAG_DATA_DIR", ".")
INDEX_PATH     = os.getenv("RAG_INDEX_PATH", os.path.join(DATA_DIR, "planville.index"))
DOCS_PATH      = os.getenv("RAG_DOCS_PATH",  os.path.join(DATA_DIR, "docs.json"))
MODEL_NAME     = os.getenv("RAG_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
AUTOBUILD      = os.getenv("RAG_AUTOBUILD", "0") in ("1", "true", "True")
MAX_DOCS       = int(os.getenv("RAG_MAX_DOCS", "100000"))  # jaga memori
SNIPPET_CHARS  = int(os.getenv("RAG_SNIPPET_CHARS", "420"))  # ringkas tiap dok
MIN_SCORE      = float(os.getenv("RAG_MIN_SCORE", "0.20"))    # drop dok yg miripnya rendah

# ========================
# 🔁 Globals (lazy cache)
# ========================
_model = None           # SentenceTransformer
_index = None           # faiss.Index
_docs: Optional[List[str]] = None
_dim: Optional[int] = None
_doc_emb: Optional[np.ndarray] = None  # dipakai hanya untuk fallback NumPy agar tidak re-encode berulang

# ========================
# 🔧 Helpers
# ========================
def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s

def _snippet(s: str, limit: int = SNIPPET_CHARS) -> str:
    """Potong per kalimat (kasar) supaya konteks padat & hemat token."""
    s = _clean_text(s)
    if len(s) <= limit:
        return s
    # coba potong di akhir kalimat terdekat
    cut = s[:limit]
    m = re.finditer(r"[.!?]\s", cut)
    last = 0
    for x in m:
        last = x.end()
    if last >= int(limit * 0.6):
        return cut[:last].strip()
    return cut.strip() + " …"

def _normalize_docs(docs: Any) -> List[str]:
    """Normalisasi isi docs → list[str]."""
    out: List[str] = []
    if isinstance(docs, list):
        for d in docs:
            if isinstance(d, str):
                out.append(_clean_text(d))
            elif isinstance(d, dict):
                t = d.get("text") or d.get("content") or d.get("body") or d.get("title")
                out.append(_clean_text(t) if isinstance(t, str) else _clean_text(json.dumps(d, ensure_ascii=False)))
            else:
                out.append(_clean_text(str(d)))
    else:
        out = [_clean_text(str(docs))]
    if len(out) > MAX_DOCS:
        out = out[:MAX_DOCS]
    return out

def _load_model():
    """Lazy load SBERT model (cached)."""
    global _model, _dim
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        vec = _model.encode(["warmup"], convert_to_numpy=True)
        _dim = int(vec.shape[1])
    return _model

def _read_docs() -> List[str]:
    """Read docs.json / docs.jsonl → list[str]."""
    global _docs
    if _docs is not None:
        return _docs
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"docs tidak ditemukan di {DOCS_PATH}")

    if DOCS_PATH.endswith(".jsonl"):
        raws = []
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    obj = line
                raws.append(obj)
        _docs = _normalize_docs(raws)
    else:
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        _docs = _normalize_docs(raw)
    return _docs

def _load_index():
    """Lazy load FAISS index (cached). Auto-build optional."""
    global _index, _dim
    if _index is not None:
        return _index
    if faiss is None:
        raise RuntimeError("FAISS tidak tersedia; fallback NumPy akan dipakai.")
    if not os.path.exists(INDEX_PATH):
        if AUTOBUILD and os.path.exists(DOCS_PATH):
            build_vector_store()
        else:
            raise FileNotFoundError(
                f"Index tidak ada di {INDEX_PATH}. Jalankan build_vector_store() atau set RAG_AUTOBUILD=1."
            )
    try:
        _index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    except Exception:
        _index = faiss.read_index(INDEX_PATH)
    if _dim is None:
        try:
            _dim = int(_index.d)  # type: ignore[attr-defined]
        except Exception:
            _dim = None
    return _index

# ========================
# 🛠 Build Vector Store
# ========================
def build_vector_store() -> None:
    """
    Bangun FAISS index dari docs, simpan ke disk.
    Embedding: SBERT (MODEL_NAME), Index: L2 (IndexFlatL2).
    """
    if faiss is None:
        raise RuntimeError("FAISS tidak tersedia—install faiss-cpu.")
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"❌ File {DOCS_PATH} tidak ditemukan.")

    docs = _read_docs()
    if not docs:
        raise ValueError("❌ Dokumen kosong / tidak valid.")

    model = _load_model()
    embeds = model.encode(docs, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    vecs = np.asarray(embeds, dtype="float32")

    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)

    os.makedirs(os.path.dirname(INDEX_PATH) or ".", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    # cache global
    global _index, _doc_emb
    _index = index
    _doc_emb = vecs  # simpan di memori jika mau, untuk keperluan lain

    print(f"✅ Index saved to {INDEX_PATH} | Total Docs: {len(docs)} | dim={vecs.shape[1]}")

# ========================
# 🔎 Query (FAISS → fallback NumPy)
# ========================
def query_index(query: str, top_k: int = 3) -> List[Tuple[float, str]]:
    """
    Return: list of (score, text) — score = 1/(1+L2) ∈ (0..1], makin besar makin mirip.
    - Pakai FAISS kalau ada; kalau tidak, brute-force NumPy.
    - Hasil sudah dipotong menjadi snippet agar hemat token (SNIPPET_CHARS).
    - Dokumen dengan score < MIN_SCORE dibuang.
    """
    if not query or not str(query).strip():
        return []

    docs = _read_docs()
    model = _load_model()

    k = int(max(1, min(top_k, len(docs))))
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    q_vec = np.asarray(q_emb, dtype="float32")  # (1, dim)

    if faiss is not None:
        index = _load_index()
        D, I = index.search(q_vec, k)  # type: ignore[attr-defined]
        d_row, i_row = D[0], I[0]
        results: List[Tuple[float, str]] = []
        for dist, idx in zip(d_row, i_row):
            if idx < 0 or idx >= len(docs):
                continue
            score = float(1.0 / (1.0 + float(dist)))
            if score < MIN_SCORE:
                continue
            results.append((score, _snippet(docs[idx])))
        return results

    # ===== Fallback NumPy =====
    global _doc_emb
    if _doc_emb is None:
        # encode semua dokumen sekali saja (cache)
        _doc_emb = model.encode(docs, convert_to_numpy=True, normalize_embeddings=False).astype("float32")

    diff = _doc_emb - q_vec  # broadcasting
    dists = np.einsum("nd,nd->n", diff, diff)  # squared L2
    idxs = np.argpartition(dists, k-1)[:k]
    idxs = idxs[np.argsort(dists[idxs])]
    results: List[Tuple[float, str]] = []
    for idx in idxs.tolist():
        dist = float(dists[idx])
        score = float(1.0 / (1.0 + dist))
        if score < MIN_SCORE:
            continue
        results.append((score, _snippet(docs[idx])))
    return results

# ========================
# 🧩 Util opsional buat main.py (kalau mau pakai langsung konteks jadi)
# ========================
def compose_context(query: str, top_k: int = 3) -> str:
    """
    Ringkas hasil menjadi satu string siap masuk ke prompt.
    Tidak dipakai langsung oleh main.py (dia menggabungkan sendiri),
    tapi berguna untuk debugging/manual test.
    """
    hits = query_index(query, top_k=top_k)
    lines = [f"- {t}" for (_, t) in hits]
    return "\n".join(lines)

# ========================
# 🚀 CLI
# ========================
if __name__ == "__main__":
    print("🔁 Building vector store from:", DOCS_PATH)
    build_vector_store()
