"""Content-addressed, bounded, persistent embedding cache.

Key   = sha256(chunk text, UTF-8).
Valid = only within one *namespace*: sha256 of the embedding model id, requested output
        dimensionality, task type and cache schema. A namespace change (e.g. another model)
        invalidates every entry. The vector dimension is recorded and enforced; vectors of a
        different dimension are never mixed.

Storage: one file, data/embedding_cache/embeddings.npz, written atomically (temp + os.replace)
and read with numpy `allow_pickle=False` (no code execution). A missing, corrupt or foreign
file is treated as an empty cache.

Bound: entries used by the build that saves the cache (the live corpus) are always kept;
other ("stale") entries are kept most-recently-used first up to MAX_STALE_BYTES
(default 64 MiB, env DOCINTEL_EMBED_CACHE_MAX_STALE_MB). Persistent size is therefore at
most live-corpus vectors + MAX_STALE_BYTES (+ 64 B key and 8 B tick per entry).
"""
import contextlib
import hashlib
import json
import os
import uuid

import numpy as np

CACHE_DIR = "data/embedding_cache"
CACHE_PATH = "data/embedding_cache/embeddings.npz"
CACHE_SCHEMA = 1


def max_stale_bytes() -> int:
    return int(float(os.environ.get("DOCINTEL_EMBED_CACHE_MAX_STALE_MB", "64")) * 1024 * 1024)


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def namespace_id(namespace: dict) -> str:
    return hashlib.sha256(json.dumps({**namespace, "cache_schema": CACHE_SCHEMA}, sort_keys=True).encode()).hexdigest()


class EmbeddingCache:
    def __init__(self, namespace: dict):
        self.namespace = namespace
        self.ns = namespace_id(namespace)
        self.dim: int | None = None
        self.tick = 0
        self._vec: dict[str, np.ndarray] = {}
        self._used_at: dict[str, int] = {}
        self.hits = self.misses = self.evicted = 0
        self.load_status = "empty"
        self.dirty = False

    # -------------------------------------------------------------- persistence
    def load(self) -> "EmbeddingCache":
        try:
            with np.load(CACHE_PATH, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                if meta.get("namespace") != self.ns or meta.get("schema") != CACHE_SCHEMA:
                    self.load_status = "invalidated_namespace"
                    self.dirty = True  # rewrite under the new namespace on save
                    return self
                keys = [k.decode() for k in z["keys"]]
                vectors = z["vectors"].astype("float32", copy=False)
                used = z["used_at"]
                if vectors.ndim != 2 or len(keys) != vectors.shape[0] or len(used) != len(keys):
                    raise ValueError("inconsistent cache arrays")
        except FileNotFoundError:
            return self
        except Exception:  # corrupt / tampered / pickled object arrays => start empty
            self.load_status = "discarded_corrupt"
            self.dirty = True
            return self
        self.dim = int(meta["dim"]) if keys else None
        self.tick = int(meta.get("tick", 0))
        self._vec = {k: vectors[i] for i, k in enumerate(keys)}
        self._used_at = {k: int(used[i]) for i, k in enumerate(keys)}
        self.load_status = "loaded"
        return self

    def save(self, live_keys: set[str]) -> dict:
        """Evict stale entries beyond the budget, then write atomically. Returns stats."""
        if self.dim is not None:
            per_entry = self.dim * 4
            stale = sorted((k for k in self._vec if k not in live_keys), key=lambda k: -self._used_at[k])
            keep_stale = max_stale_bytes() // per_entry if per_entry else 0
            for k in stale[keep_stale:]:
                del self._vec[k], self._used_at[k]
                self.evicted += 1
                self.dirty = True
        if not self.dirty:
            return self.stats()
        os.makedirs(CACHE_DIR, exist_ok=True)
        keys = list(self._vec)
        dim = self.dim or 0
        vectors = np.stack([self._vec[k] for k in keys]).astype("float32") if keys else np.zeros((0, dim), "float32")
        meta = {"schema": CACHE_SCHEMA, "namespace": self.ns, "namespace_detail": self.namespace,
                "dim": dim, "tick": self.tick}
        tmp = f"{CACHE_PATH}.tmp-{os.getpid()}-{uuid.uuid4().hex}.npz"
        try:
            with open(tmp, "wb") as f:
                np.savez(f, keys=np.array(keys, dtype="S64"), vectors=vectors,
                         used_at=np.array([self._used_at[k] for k in keys], dtype="int64"),
                         meta=np.array(json.dumps(meta)))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, CACHE_PATH)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp)
        self.dirty = False
        return self.stats()

    # -------------------------------------------------------------- access
    def begin_build(self):
        self.tick += 1

    def get(self, key: str) -> np.ndarray | None:
        v = self._vec.get(key)
        if v is None:
            self.misses += 1
            return None
        self.hits += 1
        self._used_at[key] = self.tick
        return v

    def put(self, key: str, vector) -> np.ndarray:
        v = np.asarray(vector, dtype="float32")
        if v.ndim != 1:
            raise ValueError("embedding must be a 1-D vector")
        if self.dim is None:
            self.dim = int(v.shape[0])
        elif v.shape[0] != self.dim:
            raise DimensionChanged(self.dim, int(v.shape[0]))
        self._vec[key] = v
        self._used_at[key] = self.tick
        self.dirty = True
        return v

    def reset(self):
        self._vec.clear()
        self._used_at.clear()
        self.dim = None
        self.dirty = True

    def stats(self) -> dict:
        n = len(self._vec)
        return {"entries": n, "dim": self.dim, "vector_bytes": n * (self.dim or 0) * 4,
                "hits": self.hits, "misses": self.misses, "evicted": self.evicted,
                "load_status": self.load_status}


class DimensionChanged(Exception):
    def __init__(self, cached: int, received: int):
        super().__init__(f"embedding dimension changed from {cached} to {received}")
        self.cached, self.received = cached, received
