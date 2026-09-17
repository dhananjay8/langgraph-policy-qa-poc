"""Keyword-scored paragraph retrieval over bundled policy documents.

Deliberately lightweight: documents are split into paragraph chunks and scored
by normalized term overlap so the POC has real, inspectable retrieval without
needing an embedding model or vector index.
"""

import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_docs: dict[str, str] | None = None
_chunks: list[dict] | None = None


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def load_documents() -> dict[str, str]:
    global _docs
    if _docs is None:
        _docs = {p.name: p.read_text() for p in sorted(DATA_DIR.glob("*.md"))}
    return _docs


def load_chunks() -> list[dict]:
    global _chunks
    if _chunks is None:
        _chunks = []
        for name, text in load_documents().items():
            for para in re.split(r"\n\s*\n", text):
                para = para.strip()
                if para and not para.startswith("# "):
                    _chunks.append({"doc": name, "text": para})
    return _chunks


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    q_tokens = _tokenize(question)
    if not q_tokens:
        return []
    scored = []
    for chunk in load_chunks():
        c_tokens = _tokenize(chunk["text"])
        overlap = len(q_tokens & c_tokens)
        if overlap == 0:
            continue
        score = overlap / len(q_tokens)
        scored.append({**chunk, "score": round(score, 4)})
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:top_k]


def quote_in_doc(doc: str, quote: str) -> bool:
    """Deterministic citation check: normalized quote must appear verbatim."""
    text = load_documents().get(doc)
    if text is None:
        return False
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    return norm(quote) in norm(text)
