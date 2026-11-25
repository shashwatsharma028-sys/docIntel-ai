import os
import threading
from collections import OrderedDict

import numpy as np
from dotenv import load_dotenv
from google import genai

from backend import index_store

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

INDEX_PATH = index_store.INDEX_PATH
CHUNKS_PATH = index_store.CHUNKS_PATH

# ---------------------------------------------------------------------------------------------
# Version-aware generation cache (P1B).
#
# Each entry is one immutable index_store.Generation (FAISS index + chunk metadata loaded
# together under the shared publish lock), keyed by the real path of the index directory, so
# different working directories never share an entry. On every query the entry is validated
# against index_store.signature() -- the (device, inode, size, mtime_ns) identities of
# manifest.json, faiss.index and chunks.pkl -- which changes on every publish (os.replace
# gives new inodes). A mismatch triggers exactly one reload; concurrent readers wait for it.
#
# Memory bound: at most MAX_CACHED_GENERATIONS entries (LRU across directories); each holds
# ntotal * d * 4 bytes of vectors plus chunk text. While a reload is in progress the previous
# generation stays alive until in-flight queries finish, so peak memory is ~2x one generation.
# ---------------------------------------------------------------------------------------------
MAX_CACHED_GENERATIONS = int(os.environ.get("DOCINTEL_RETRIEVER_CACHE_GENERATIONS", "2"))
_cache: "OrderedDict[str, index_store.Generation]" = OrderedDict()
_cache_lock = threading.Lock()
_load_lock = threading.Lock()
STATS = {"hits": 0, "loads": 0, "reloads": 0, "evictions": 0}


def get_embedding(text: str) -> list:
    result = client.models.embed_content(model="gemini-embedding-001", contents=text)
    return result.embeddings[0].values


def _lookup(key):
    gen = _cache.get(key[0])
    if gen is not None and gen.key == key:
        _cache.move_to_end(key[0])
        STATS["hits"] += 1
        return gen
    return None


def current_generation() -> index_store.Generation:
    """Return the cached generation if it is still the published one, else load it once."""
    key = index_store.signature()  # FileNotFoundError when nothing is published (as before)
    with _cache_lock:
        gen = _lookup(key)
    if gen is not None:
        return gen
    with _load_lock:
        key = index_store.signature()
        with _cache_lock:
            gen = _lookup(key)
        if gen is not None:
            return gen
        gen = index_store.load_generation()
        with _cache_lock:
            if gen.key[0] in _cache:
                STATS["reloads"] += 1
            _cache[gen.key[0]] = gen
            _cache.move_to_end(gen.key[0])
            while len(_cache) > max(1, MAX_CACHED_GENERATIONS):
                _cache.popitem(last=False)
                STATS["evictions"] += 1
            STATS["loads"] += 1
        return gen


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def cache_info() -> dict:
    with _cache_lock:
        entries = [{"directory": d, "generation": g.generation_id, "n_chunks": len(g.chunks), "dim": g.index.d,
                    "resident_bytes": g.nbytes, "load_ms": round(g.load_ms, 3)} for d, g in _cache.items()]
    return {"entries": entries, "max_entries": MAX_CACHED_GENERATIONS, **STATS}


# ---------------------------------------------------------------------------------------------
# Optional cross-encoder reranking (DOCINTEL_RERANKER=minilm; default off = original dense ranking).
# The top DOCINTEL_RERANK_POOL dense candidates are rescored by backend.reranker; on any reranker
# failure the dense order is returned (counted in STATS["rerank_failures"]).
# ---------------------------------------------------------------------------------------------
_RERANKER = None
_reranker_lock = threading.Lock()
STATS.update(reranked=0, rerank_failures=0)


def reranker_mode() -> str:
    mode = os.environ.get("DOCINTEL_RERANKER", "off")
    if mode not in ("off", "minilm"):
        raise ValueError(f"DOCINTEL_RERANKER must be 'off' or 'minilm', got {mode!r}")
    return mode


def rerank_pool() -> int:
    n = int(os.environ.get("DOCINTEL_RERANK_POOL", "20"))
    if not 1 <= n <= 100:
        raise ValueError(f"DOCINTEL_RERANK_POOL must be in 1..100, got {n}")
    return n


def _get_reranker():
    global _RERANKER
    with _reranker_lock:
        if _RERANKER is None:
            from backend.reranker import MiniLMReranker
            _RERANKER = MiniLMReranker()
        return _RERANKER


def _retrieve_reranked(gen, query, top_k):
    index, chunks = gen.index, gen.chunks
    query_vec = np.array([get_embedding(query)]).astype("float32")
    pool = max(top_k, rerank_pool())
    _, indices = index.search(query_vec, pool)
    cand = [int(i) for i in indices[0] if 0 <= i < len(chunks)]
    try:
        scores = _get_reranker().score(query, [chunks[i][1] for i in cand])
        cand = [cand[j] for j in np.argsort(-np.asarray(scores), kind="stable")]
        STATS["reranked"] += 1
    except Exception as e:  # availability: fall back to the dense order
        STATS["rerank_failures"] += 1
        print(f"Reranker failed ({type(e).__name__}: {e}); using dense order")
    return [{"source": chunks[i][0], "text": chunks[i][1]} for i in cand[:top_k]]


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    gen = current_generation()
    if reranker_mode() == "minilm":
        return _retrieve_reranked(gen, query, top_k)
    index, chunks = gen.index, gen.chunks
    query_vec = np.array([get_embedding(query)]).astype("float32")
    _, indices = index.search(query_vec, top_k)
    return [{"source": chunks[i][0], "text": chunks[i][1]} for i in indices[0] if 0 <= i < len(chunks)]
