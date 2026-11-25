import hashlib
import json
import os
import time
import fitz
import numpy as np
from dotenv import load_dotenv
from google import genai

from backend import index_store, structure_chunker
from backend.pdf_runtime import run_pdf
from backend.embedding_cache import DimensionChanged, EmbeddingCache, text_key

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

INDEX_PATH = index_store.INDEX_PATH
CHUNKS_PATH = index_store.CHUNKS_PATH
CHUNK_SIZE = 800      # was 500
CHUNK_OVERLAP = 100   # was 50
MIN_CHUNK_CHARS = 50
CHUNKER_VERSION = 1
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_OUTPUT_DIMENSIONALITY = None  # provider default (not requested explicitly)
EMBEDDING_TASK_TYPE = None              # provider default

MAX_EMBED_BATCH = 100  # upper bound accepted from configuration; verify against current provider limits


class EmbeddingResponseError(RuntimeError):
    """A provider response failed validation (count, dimension or non-finite values)."""


def embed_batch_size() -> int:
    n = int(os.environ.get("DOCINTEL_EMBED_BATCH_SIZE", "1"))
    if not 1 <= n <= MAX_EMBED_BATCH:
        raise ValueError(f"DOCINTEL_EMBED_BATCH_SIZE must be in 1..{MAX_EMBED_BATCH}, got {n}")
    return n


def get_embeddings(texts: list[str]) -> list:
    """One provider request for several independent inputs; validated before use."""
    result = client.models.embed_content(model=EMBEDDING_MODEL, contents=list(texts))
    values = [e.values for e in (result.embeddings or [])]
    if len(values) != len(texts):
        raise EmbeddingResponseError(f"expected {len(texts)} embeddings, got {len(values)}")
    arr = np.asarray(values, dtype="float32")
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise EmbeddingResponseError("embeddings must be finite and of equal dimension")
    return list(arr)


def get_embedding(text: str) -> list:
    result = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    return result.embeddings[0].values

def parse_pdf(pdf_path: "str | bytes") -> str:
    """Extract text from a PDF path or bytes; runs on the dedicated PyMuPDF thread."""
    return run_pdf(_parse_pdf, pdf_path)


def _parse_pdf(pdf):
    doc = fitz.open(stream=pdf, filetype="pdf") if isinstance(pdf, (bytes, bytearray)) else fitz.open(pdf)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c.strip()) > MIN_CHUNK_CHARS]


def embedding_namespace() -> dict:
    """Everything that determines an embedding besides the text itself."""
    return {"provider": "google-genai", "model": EMBEDDING_MODEL,
            "output_dimensionality": EMBEDDING_OUTPUT_DIMENSIONALITY, "task_type": EMBEDDING_TASK_TYPE}


def chunking_strategy() -> str:
    """'words' (default, original 800/100-word windows) or 'structure' (opt-in, see structure_chunker)."""
    strategy = os.environ.get("DOCINTEL_CHUNKING", "words")
    if strategy not in ("words", "structure"):
        raise ValueError(f"unknown DOCINTEL_CHUNKING={strategy!r}")
    return strategy


def chunker_signature() -> str:
    if chunking_strategy() == "structure":
        return f"structure:budget={structure_chunker.TOKEN_BUDGET}:est=wordpiece7:v{CHUNKER_VERSION}"
    return f"words:size={CHUNK_SIZE}:overlap={CHUNK_OVERLAP}:min_chars={MIN_CHUNK_CHARS}:v{CHUNKER_VERSION}"


def chunk_document(path: str, data: "bytes | None" = None) -> list[str]:
    """Chunk one document; with `data`, the given bytes snapshot is used instead of re-reading `path`."""
    src = path if data is None else data
    if chunking_strategy() == "structure":
        return structure_chunker.chunk_pdf_structured(src)
    return chunk_text(parse_pdf(src))


def replace_and_build(staged: str, path: str, pdf_folder: str = "data/pdfs") -> dict:
    """Atomically move a validated upload into the corpus and rebuild under ONE build-lock hold,
    so no other writer can change the corpus between replacement, fingerprinting and parsing."""
    t_start = time.perf_counter()
    stats = {"documents": 0, "documents_parsed": 0, "documents_reused": 0, "chunks": 0,
             "embeddings_computed": 0, "cache_hits": 0, "published": False, "unchanged": False}
    with index_store.build_lock():
        os.replace(staged, path)
        return _build_index_locked(pdf_folder, t_start, stats)


def _corpus_fingerprint(chunks, namespace) -> str:
    h = hashlib.sha256(json.dumps(namespace, sort_keys=True).encode())
    for source, text in chunks:
        h.update(b"\0" + source.encode() + b"\0" + text_key(text).encode())
    return h.hexdigest()


def build_index(pdf_folder: str = "data/pdfs") -> dict:
    """Incrementally (re)build the index for every *.pdf in `pdf_folder`.

    * Documents are fingerprinted (sha256 of file bytes + chunker signature); an unchanged
      document reuses its chunk texts from the published generation without re-parsing.
    * Chunks are embedded only on an embedding-cache miss (content-addressed, namespaced).
    * The FAISS IndexFlatL2 is then REBUILT from cached + new vectors and published atomically.
      This is incremental *embedding* reuse, not incremental FAISS insertion.
    * On any failure the previously published generation stays in place; embeddings obtained
      before the failure are kept in the cache so a retry does not repeat them.
    Returns build statistics (also used by benchmarks); callers may ignore it.
    """
    t_start = time.perf_counter()
    stats = {"documents": 0, "documents_parsed": 0, "documents_reused": 0, "chunks": 0,
             "embeddings_computed": 0, "cache_hits": 0, "published": False, "unchanged": False}
    with index_store.build_lock():
        return _build_index_locked(pdf_folder, t_start, stats)


def _build_index_locked(pdf_folder, t_start, stats) -> dict:
    """Body of build_index; the caller holds index_store.build_lock()."""
    filenames = sorted(f for f in os.listdir(pdf_folder) if f.endswith(".pdf"))
    previous = index_store.load_metadata()
    reusable = {}
    if previous is not None:
        prev_manifest, prev_chunks = previous
        if prev_manifest.get("chunker") == chunker_signature():
            offset = 0
            for doc in prev_manifest["documents"]:
                n = doc["n_chunks"]
                reusable[doc["doc_key"]] = [t for _, t in prev_chunks[offset:offset + n]]
                offset += n

    all_chunks, documents = [], []
    t_parse = time.perf_counter()
    for filename in filenames:
        path = os.path.join(pdf_folder, filename)
        with open(path, "rb") as f:
            data = f.read()  # one snapshot: the same bytes are fingerprinted AND parsed
        file_sha = hashlib.sha256(data).hexdigest()
        doc_key = hashlib.sha256(f"{file_sha}|{chunker_signature()}".encode()).hexdigest()
        texts = reusable.get(doc_key)
        if texts is None:
            print(f"Processing: {filename}")
            texts = chunk_document(path, data)
            stats["documents_parsed"] += 1
        else:
            stats["documents_reused"] += 1
        documents.append({"source": filename, "file_sha256": file_sha, "doc_key": doc_key,
                          "n_chunks": len(texts)})
        all_chunks.extend((filename, t) for t in texts)
    stats["parse_ms"] = (time.perf_counter() - t_parse) * 1000
    stats["documents"] = len(filenames)
    stats["chunks"] = len(all_chunks)

    if not all_chunks:
        print("No PDFs found.")
        # D-09: an empty corpus must not leave a previous generation searchable.
        stats["published"] = index_store.publish_empty()
        stats["total_ms"] = (time.perf_counter() - t_start) * 1000
        return stats

    namespace = embedding_namespace()
    fingerprint = _corpus_fingerprint(all_chunks, namespace)
    if previous is not None and previous[0].get("corpus_fingerprint") == fingerprint:
        stats["unchanged"] = True  # identical corpus: keep the published generation (no invalidation)
        stats["total_ms"] = (time.perf_counter() - t_start) * 1000
        print(f"Done! {len(all_chunks)} chunks indexed (unchanged).")
        return stats

    cache = EmbeddingCache(namespace).load()
    cache.begin_build()
    t_embed = time.perf_counter()
    try:
        try:
            vectors = _embed_all(all_chunks, cache, stats)
        except DimensionChanged:
            # The provider now returns a different dimension: cached vectors are invalid.
            cache.reset()
            stats["cache_hits"] = 0  # hits before the restart were served from invalid vectors
            vectors = _embed_all(all_chunks, cache, stats)
    finally:
        # Persist whatever was embedded, even if the build failed part-way.
        stats["cache"] = cache.save({text_key(t) for _, t in all_chunks})
    stats["embed_ms"] = (time.perf_counter() - t_embed) * 1000

    manifest = index_store.publish(all_chunks, np.vstack(vectors), documents,
                                   {**namespace, "dim": int(vectors[0].shape[0])}, fingerprint,
                                   chunker=chunker_signature())
    stats.update(published=True, generation=manifest["generation"],
                 index_construct_ms=manifest["_timings_ms"]["index_construct"],
                 publish_ms=manifest["_timings_ms"]["publish_total"])
    stats["total_ms"] = (time.perf_counter() - t_start) * 1000
    print(f"Done! {len(all_chunks)} chunks indexed.")
    return stats


def _embed_all(all_chunks, cache, stats) -> list:
    if embed_batch_size() > 1:
        return _embed_all_batched(all_chunks, cache, stats, embed_batch_size())
    vectors = []
    for _, text in all_chunks:
        key = text_key(text)
        vec = cache.get(key)
        if vec is None:
            raw = get_embedding(text)
            stats["embeddings_computed"] += 1  # count every provider invocation, even if rejected below
            stats["embedding_requests"] = stats.get("embedding_requests", 0) + 1
            vec = cache.put(key, raw)
        else:
            stats["cache_hits"] += 1
        vectors.append(vec.reshape(1, -1))
    return vectors


def _embed_all_batched(all_chunks, cache, stats, batch_size, retries=2):
    """Unique cache misses (first-seen order) are embedded in bounded batches; each batch is
    validated and cached before the next, so a later failure keeps earlier work and publishes nothing."""
    keys = [text_key(t) for _, t in all_chunks]
    texts = dict(zip(keys, (t for _, t in all_chunks)))
    resolved, misses = {}, []
    for k in dict.fromkeys(keys):  # unique, first-seen order
        v = cache.get(k)
        if v is None:
            misses.append(k)
        else:
            resolved[k] = v
    stats["cache_hits"] += len(keys) - len(misses)  # same definition as the single-item path
    for i in range(0, len(misses), batch_size):
        batch = misses[i:i + batch_size]
        for attempt in range(retries + 1):
            stats["embedding_requests"] = stats.get("embedding_requests", 0) + 1
            try:
                vecs = get_embeddings([texts[k] for k in batch])
                break
            except DimensionChanged:
                raise
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(0.5 * 2 ** attempt)
        stats["embeddings_computed"] += len(batch)
        for k, v in zip(batch, vecs):
            resolved[k] = cache.put(k, v)
    return [resolved[k].reshape(1, -1) for k in keys]


if __name__ == "__main__":
    build_index()
