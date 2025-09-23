# rag_engine.py — fast, cached, score-aware (L2 → 0..1) — 2025-09-23
from __future__ import annotations
import os, json
from typing import List, Tuple, Any, Optional

import numpy as np

# FAISS bisa nggak ada di beberapa env; tapi di Railway biasa ada.
try:
    import faiss  # type: ignore
except Exception as _e:
    faiss = None  # noqa: F401

# ========================
# 📂 Paths & Model (ENV-aware)
# ========================
DATA_DIR      = os.getenv("RAG_DATA_DIR", ".")
INDEX_PATH    = os.getenv("RAG_INDEX_PATH", os.path.join(DATA_DIR, "planville.index"))
DOCS_PATH     = os.getenv("RAG_DOCS_PATH",  os.path.join(DATA_DIR, "docs.json"))
MODEL_NAME    = os.getenv("RAG_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
AUTOBUILD     = os.getenv("RAG_AUTOBUILD", "0") in ("1", "true", "True")

# ========================
# 🔁 Globals (lazy cached)
# ========================
_model = None          # SentenceTransformer
_index = None          # faiss.Index
_docs: Optional[List[str]] = None
_dim: Optional[int] = None

def _load_model():
    """Lazy load SBERT model (cached)."""
    global _model, _dim
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        # warm a tiny encode to ensure dim known:
        vec = _model.encode(["warmup"], convert_to_numpy=True)
        _dim = int(vec.shape[1])
    return _model

def _read_docs() -> List[str]:
    """Read docs.json yang sudah dinormalisasi ke list[str]."""
    global _docs
    if _docs is not None:
        return _docs
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"docs.json tidak ditemukan di {DOCS_PATH}")
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Pastikan list[str]
    if isinstance(raw, list):
        out: List[str] = []
        for d in raw:
            if isinstance(d, str):
                out.append(d)
            elif isinstance(d, dict):
                t = d.get("text") or d.get("content") or d.get("body") or d.get("title")
                out.append(t if isinstance(t, str) else json.dumps(d, ensure_ascii=False))
            else:
                out.append(str(d))
        _docs = out
    else:
        _docs = [str(raw)]
    return _docs

def _load_index():
    """Lazy load FAISS index (cached). Auto-build optional."""
    global _index, _dim
    if _index is not None:
        return _index
    if faiss is None:
        raise RuntimeError("FAISS tidak tersedia di environment.")
    if not os.path.exists(INDEX_PATH):
        if AUTOBUILD and os.path.exists(DOCS_PATH):
            build_vector_store()  # akan set _index & _docs
        else:
            raise FileNotFoundError(
                f"Index tidak ditemukan di {INDEX_PATH}. "
                f"Jalankan build_vector_store() atau set RAG_AUTOBUILD=1."
            )
    # coba mmap (hemat memori); fallback ke read biasa
    try:
        _index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    except Exception:
        _index = faiss.read_index(INDEX_PATH)
    # dim bisa didapat dari index
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
    """Bangun FAISS index dari docs.json, simpan ke planville.index, dan normalkan docs ke list[str]."""
    if faiss is None:
        raise RuntimeError("FAISS tidak tersedia—install faiss-cpu.")
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"❌ File {DOCS_PATH} tidak ditemukan.")

    # baca & normalisasi
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)

    texts = _normalize_docs(raw_docs)
    if not texts:
        raise ValueError("❌ docs.json kosong / tidak valid.")

    # embed → float32
    model = _load_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    vecs = np.asarray(embeddings, dtype="float32")

    # build index (L2)
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)

    # simpan index + docs normal
    os.makedirs(os.path.dirname(INDEX_PATH) or ".", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    # set cache global
    global _index, _docs, _dim
    _index = index
    _docs  = texts
    _dim   = vecs.shape[1]

    print(f"✅ Index saved to {INDEX_PATH} | Total Docs: {len(texts)} | dim={_dim}")

def _normalize_docs(docs: Any) -> List[str]:
    """Normalisasi isi docs → list[str]."""
    out: List[str] = []
    if isinstance(docs, list):
        for d in docs:
            if isinstance(d, str):
                out.append(d)
            elif isinstance(d, dict):
                t = d.get("text") or d.get("content") or d.get("body") or d.get("title")
                out.append(t if isinstance(t, str) else json.dumps(d, ensure_ascii=False))
            else:
                out.append(str(d))
    else:
        out = [str(docs)]
    return out

# ========================
# 🔎 Query
# ========================
def query_index(query: str, top_k: int = 3) -> List[Tuple[float, str]]:
    """
    Return: list of (score, text)
    score = 1 / (1 + L2)  → 0..1 (semakin besar = semakin mirip)
    """
    if not query or not str(query).strip():
        return []

    index = _load_index()
    docs  = _read_docs()
    model = _load_model()

    k = int(max(1, min(top_k, len(docs))))
    q_emb = model.encode([query], convert_to_numpy=True)
    q_vec = np.asarray(q_emb, dtype="float32")

    D, I = index.search(q_vec, k)  # type: ignore[attr-defined]
    d_row, i_row = D[0], I[0]

    results: List[Tuple[float, str]] = []
    for dist, idx in zip(d_row, i_row):
        if idx < 0 or idx >= len(docs):
            continue
        score = float(1.0 / (1.0 + float(dist)))
        text  = docs[idx] if isinstance(docs[idx], str) else json.dumps(docs[idx], ensure_ascii=False)
        results.append((score, text))
    return results

# ========================
# 🚀 CLI
# ========================
if __name__ == "__main__":
    print("🔁 Building vector store from:", DOCS_PATH)
    build_vector_store()
