# prepare_docs.py — bersih-bersih + chunking docs.json → docs.clean.json
import json, re, unicodedata, os
from typing import List

SRC  = os.getenv("SRC", "docs.json")
DEST = os.getenv("DEST", "docs.clean.json")

# Kata hubung Jerman umum: bantu “menyisipkan spasi” kalau nempel
GER_PREP = r"(?:und|oder|für|mit|ohne|von|vom|zum|zur|am|im|an|auf|in|aus|bei|durch|gegen|ohne|um|über|unter|nach|seit)"
SPACE_FIX_PATTERNS = [
    (re.compile(r"([a-zäöüß])([A-ZÄÖÜ])"), r"\1 \2"),              # camel-like: xY → x Y
    (re.compile(rf"([a-zäöüß])({GER_PREP})([A-ZÄÖÜa-zäöüß])"), r"\1 \2 \3"),  # xfürY → x für Y
    (re.compile(r"(\w)([–—-])(\w)"), r"\1 – \3"),                   # tambah spasi sekitar dash
]

# Perbaikan eksplisit frasa yang sering nempel
EXPLICIT_FIXES = {
    "Ingenieurbürofürenergetische": "Ingenieurbüro für energetische",
    "Energiewendevoran": "Energiewende voran",
    "passgenaue Photovoltaik-Lösungen": "passgenaue Photovoltaik-Lösungen",
    "rundumsorgenfreies": "rundum sorgenfreies",
    "HausbesitzerPhotovoltaikWärmepumpenDachsanierungFenster":
        "Hausbesitzer, Photovoltaik, Wärmepumpen, Dachsanierung, Fenster",
    "Wunschtermin.": "Wunschtermin.",
}

DROP_HINTS = [
    "Newsletter", "Melden Sie sich", "Bleiben Sie auf dem Laufenden",
    "Betreff", "Ich akzeptiere die Datenschutzbedingungen",
    "Notfall-Nummer", "07147 /", "Gerhard-Welter-Straße",
    "Fachbereichsleiter", "Teamleiter", "Grafikdesignerin",
]

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    # Hapus emoji & kontrol
    s = "".join(ch for ch in s if (ord(ch) >= 32 and not (0x1F300 <= ord(ch) <= 0x1FAFF)))
    # Perbaiki eksplisit
    for a,b in EXPLICIT_FIXES.items():
        s = s.replace(a, b)
    # Spasi ganda → tunggal
    s = re.sub(r"\s+", " ", s).strip()
    # Heuristik spasi
    tmp = s
    for pat, rep in SPACE_FIX_PATTERNS:
        tmp = pat.sub(rep, tmp)
    # Titik dua spasi setelah tanda baca
    tmp = re.sub(r"([,;:])([^\s])", r"\1 \2", tmp)
    return tmp

def is_informative(s: str) -> bool:
    if len(s) < 40:
        return False
    if any(h.lower() in s.lower() for h in DROP_HINTS):
        return False
    return True

def dedup_keep_order(lines: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in lines:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out

def chunk_sentences(text: str, max_chars: int = 900) -> List[str]:
    # Split per kalimat kasar
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks, buf = [], ""
    for p in parts:
        if not p: continue
        # batasi kalimat terlalu panjang
        p = p.strip()
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf: chunks.append(buf)
            buf = p
    if buf: chunks.append(buf)
    return chunks

def main():
    raw = json.load(open(SRC, "r", encoding="utf-8"))
    if isinstance(raw, list):
        items = [str(x) for x in raw]
    else:
        items = [str(raw)]

    # 1) Normalisasi dan drop noise
    cleaned = []
    for s in items:
        t = normalize_text(s)
        if is_informative(t):
            cleaned.append(t)

    # 2) Gabungkan kalimat yang se-topik (heuristik: join semua lalu re-chunk)
    # (opsi sederhana: join jadi satu korpus lalu chunk)
    corpus = " ".join(cleaned)
    corpus = re.sub(r"\s+", " ", corpus).strip()

    # 3) Chunk 600–900 chars, lalu de-dup lagi (barangkali chunk awal identik)
    chunks = chunk_sentences(corpus, max_chars=900)
    chunks = [c.strip() for c in chunks if len(c.strip()) >= 120]
    chunks = dedup_keep_order(chunks)

    # 4) Simpan
    json.dump(chunks, open(DEST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"✅ Cleaned: {len(cleaned)} items → Chunked: {len(chunks)}")
    print(f"→ Output: {DEST}")

if __name__ == "__main__":
    main()
