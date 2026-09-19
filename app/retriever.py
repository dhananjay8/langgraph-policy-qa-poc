"""Hybrid retrieval over bundled policy documents.

Combines semantic search (Azure OpenAI embeddings + FAISS) with keyword
scoring.  Falls back to keyword-only retrieval when embeddings are
unavailable (missing env vars, import errors, etc.) so the POC keeps
working without a vector index.

Chunks carry section-heading metadata for richer context in the generate
prompt.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

DATA_DIR = Path(__file__).parent / "data"

log = logging.getLogger("policy-qa.retriever")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HEADING_RE = re.compile(r"^##\s+(.+)", re.MULTILINE)
_WS_RE = re.compile(r"\s+")

_docs: dict[str, str] | None = None
_chunks: list[dict] | None = None
_faiss_index: Any = None
_chunk_embeddings: np.ndarray | None = None
_embedding_dim: int = 0
_embed_client: Any = None

# ---------------------------------------------------------------------------
# Document & chunk loading
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def load_documents() -> dict[str, str]:
    global _docs
    if _docs is None:
        _docs = {p.name: p.read_text() for p in sorted(DATA_DIR.glob("*.md"))}
    return _docs


def _current_section(text: str, pos: int) -> str:
    """Return the nearest ## heading above *pos* in *text*."""
    best = ""
    for m in _HEADING_RE.finditer(text):
        if m.start() <= pos:
            best = m.group(1).strip()
    return best


def load_chunks() -> list[dict]:
    """Split documents into paragraph chunks with section metadata."""
    global _chunks
    if _chunks is not None:
        return _chunks
    _chunks = []
    for name, text in load_documents().items():
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para or para.startswith("# "):
                continue
            section = _current_section(text, text.find(para))
            _chunks.append({"doc": name, "section": section, "text": para})
    return _chunks

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _get_embed_client() -> Any:
    """Singleton Azure OpenAI client for embeddings."""
    global _embed_client
    if _embed_client is not None:
        return _embed_client
    try:
        from openai import AzureOpenAI
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        api_ver = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
        if not endpoint or not api_key:
            return None
        _embed_client = AzureOpenAI(
            azure_endpoint=endpoint, api_key=api_key, api_version=api_ver,
        )
        return _embed_client
    except Exception:
        return None


def _get_embeddings(texts: list[str]) -> np.ndarray | None:
    """Call Azure OpenAI embeddings endpoint.  Returns None on any failure."""
    try:
        client = _get_embed_client()
        if client is None:
            return None
        deploy = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")
        resp = client.embeddings.create(model=deploy, input=texts)
        vecs = [d.embedding for d in resp.data]
        return np.array(vecs, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        log.warning("Embedding call failed, falling back to keyword retrieval: %s", exc)
        return None


def _build_faiss_index() -> bool:
    """Build a FAISS flat-IP index over chunk embeddings.  Returns success."""
    global _faiss_index, _chunk_embeddings, _embedding_dim
    if _faiss_index is not None:
        return True
    try:
        import faiss  # type: ignore[import-untyped]
    except ImportError:
        log.info("faiss-cpu not installed; using keyword-only retrieval")
        return False

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    vecs = _get_embeddings(texts)
    if vecs is None:
        return False

    # L2-normalize for cosine similarity via inner-product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vecs = vecs / norms

    _embedding_dim = vecs.shape[1]
    _faiss_index = faiss.IndexFlatIP(_embedding_dim)
    _faiss_index.add(vecs)
    _chunk_embeddings = vecs
    log.info("FAISS index built: %d chunks, dim=%d", len(chunks), _embedding_dim)
    return True

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _keyword_retrieve(question: str, top_k: int) -> list[dict]:
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


def _semantic_retrieve(question: str, top_k: int) -> list[dict]:
    q_vec = _get_embeddings([question])
    if q_vec is None or _faiss_index is None:
        return []
    norm = np.linalg.norm(q_vec)
    if norm > 0:
        q_vec = q_vec / norm
    scores, indices = _faiss_index.search(q_vec, top_k)
    chunks = load_chunks()
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append({**chunks[idx], "score": round(float(score), 4)})
    return results


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    """Hybrid retrieve: merge semantic + keyword results, re-ranked by combined score."""
    use_semantic = _build_faiss_index()

    keyword_results = _keyword_retrieve(question, top_k=top_k * 2)
    semantic_results = _semantic_retrieve(question, top_k=top_k * 2) if use_semantic else []

    # Merge by (doc, text) key — combine scores with weighting
    seen: dict[tuple[str, str], dict] = {}
    for r in semantic_results:
        key = (r["doc"], r["text"])
        seen[key] = {**r, "score": round(r["score"] * 0.7, 4)}  # semantic weight
    for r in keyword_results:
        key = (r["doc"], r["text"])
        if key in seen:
            seen[key]["score"] = round(seen[key]["score"] + r["score"] * 0.3, 4)
        else:
            seen[key] = {**r, "score": round(r["score"] * 0.3, 4)}  # keyword weight

    merged = sorted(seen.values(), key=lambda c: c["score"], reverse=True)
    return merged[:top_k]


def normalize_ws(s: str) -> str:
    """Collapse whitespace to single spaces and strip."""
    return _WS_RE.sub(" ", s).strip()


def quote_in_doc(doc: str, quote: str) -> bool:
    """Deterministic citation check: normalized quote must appear verbatim."""
    text = load_documents().get(doc)
    if text is None:
        return False
    return normalize_ws(quote) in normalize_ws(text)
