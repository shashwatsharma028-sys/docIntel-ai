import hashlib
import json
import os
import time
import fitz
import numpy as np
from dotenv import load_dotenv
from google import genai

from backend import index_store
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

def get_embedding(text: str) -> list:
    result = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    return result.embeddings[0].values

def parse_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
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


def chunker_signature() -> str:
    return f"words:size={CHUNK_SIZE}:overlap={CHUNK_OVERLAP}:min_chars={MIN_CHUNK_CHARS}:v{CHUNKER_VERSION}"


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
                file_sha = hashlib.sha256(f.read()).hexdigest()
            doc_key = hashlib.sha256(f"{file_sha}|{chunker_signature()}".encode()).hexdigest()
            texts = reusable.get(doc_key)
            if texts is None:
                print(f"Processing: {filename}")
                texts = chunk_text(parse_pdf(path))
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
    vectors = []
    for _, text in all_chunks:
        key = text_key(text)
        vec = cache.get(key)
        if vec is None:
            raw = get_embedding(text)
            stats["embeddings_computed"] += 1  # count every provider invocation, even if rejected below
            vec = cache.put(key, raw)
        else:
            stats["cache_hits"] += 1
        vectors.append(vec.reshape(1, -1))
    return vectors


if __name__ == "__main__":
    build_index()
