# rag_engine.py  — Opsi B (root paths) + score-aware results
import os
import json
import faiss
import numpy as np
from typing import List, Tuple, Any
from sentence_transformers import SentenceTransformer

# ========================
# 📂 Path & Model Settings
# ========================
# ▶ Opsi B: simpan docs.json & planville.index di ROOT repo
DATA_DIR   = "."
INDEX_PATH = os.path.join(DATA_DIR, "planville.index")
DOCS_PATH  = os.path.join(DATA_DIR, "docs.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ========================
# 🧠 Load Embedding Model
# ========================
model = SentenceTransformer(MODEL_NAME)

def _extract_texts(docs: Any) -> List[str]:
    """
    Normalisasi isi docs → list[str]
    - Jika docs sudah list[str] → pakai apa adanya.
    - Jika list[dict], ambil key umum: text/content/body/title (fallback ke str(dict)).
    """
    texts: List[str] = []
    if isinstance(docs, list):
        for d in docs:
            if isinstance(d, str):
                texts.append(d)
            elif isinstance(d, dict):
                t = d.get("text") or d.get("content") or d.get("body") or d.get("title")
                texts.append(t if isinstance(t, str) else json.dumps(d, ensure_ascii=False))
            else:
                texts.append(str(d))
    else:
        # kalau bukan list, paksa string
        texts = [str(docs)]
    return texts

# ========================
# 🔧 Build Vector Store
# ========================
def build_vector_store() -> None:
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"❌ File {DOCS_PATH} tidak ditemukan. Taruh docs.json di ROOT atau sesuaikan path.")

    with open(DOCS_PATH, encoding="utf-8") as f:
        raw_docs = json.load(f)

    texts = _extract_texts(raw_docs)
    if not texts:
        raise ValueError("❌ docs.json kosong atau tidak valid.")

    # ➜ FAISS butuh float32
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    vecs = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)

    # Simpan index + (opsional) normalisasi ulang docs agar konsisten
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    print(f"✅ Index saved to {INDEX_PATH} | Total Docs: {len(texts)}")

# ========================
# 🔍 Query Vector Index
# ========================
def query_index(query: str, top_k: int = 3) -> List[Tuple[float, str]]:
    """
    Return: list of (score, text)
    - score = 1 / (1 + L2 distance)  → 0..1 (semakin besar = semakin mirip)
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        raise FileNotFoundError("❌ Index atau dokumen belum dibangun. Jalankan build_vector_store() dulu.")

    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, encoding="utf-8") as f:
        docs = json.load(f)  # <- sudah dinormalisasi ke list[str] saat build

    q_emb = model.encode([query], convert_to_numpy=True)
    q_vec = np.asarray(q_emb, dtype="float32")
    D, I = index.search(q_vec, top_k)

    results: List[Tuple[float, str]] = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        score = float(1.0 / (1.0 + float(dist)))  # konversi jarak → skor 0..1
        text = docs[idx] if isinstance(docs[idx], str) else json.dumps(docs[idx], ensure_ascii=False)
        results.append((score, text))
    return results

# ========================
# 🚀 CLI Mode
# ========================
if __name__ == "__main__":
    print("🔁 Building vector store from ROOT/docs.json ...")
    build_vector_store()
