"""Index generation storage: atomic publication, coherent loading, safe metadata.

On-disk layout (backward compatible with the original implementation):

    data/index/faiss.index     FAISS IndexFlatL2                  (unchanged format)
    data/index/chunks.pkl      list[(source, text)] -- primitives  (unchanged format)
    data/index/manifest.json   NEW, additive: generation id, artifact identities and
                               sha256, per-document fingerprints, embedding namespace

A *generation* is one coherent (faiss.index, chunks.pkl, manifest.json) triple.

Writers build every artifact in a temporary file, then swap all three into place with
os.replace while holding an exclusive publish lock. Readers that load while holding the
shared publish lock therefore never observe a half-published generation. If a process
dies between renames, the manifest's recorded artifact identities no longer match the
files and loading fails with IndexIntegrityError instead of returning misaligned data;
the next successful build repairs it.

chunks.pkl is never unpickled with the standard loader: `loads_chunks` refuses every
global (so no callable can be reached and no code can run) and validates the schema.
"""
import contextlib
import hashlib
import io
import json
import os
import pickle
import threading
import time
import uuid

import faiss
import numpy as np

try:  # POSIX advisory locks; on other platforms only in-process locking is available.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

INDEX_DIR = "data/index"
INDEX_PATH = "data/index/faiss.index"
CHUNKS_PATH = "data/index/chunks.pkl"
MANIFEST_PATH = "data/index/manifest.json"
LOCK_DIR = "data"
BUILD_LOCK = ".index-build.lock"
PUBLISH_LOCK = ".index-publish.lock"
MANIFEST_SCHEMA = 1

_fallback_locks = {BUILD_LOCK: threading.RLock(), PUBLISH_LOCK: threading.RLock()}
COUNTERS = {"index_disk_loads": 0, "metadata_disk_loads": 0, "publishes": 0, "integrity_failures": 0}


class IndexIntegrityError(RuntimeError):
    """The files on disk do not form one coherent, valid generation."""


# ----------------------------------------------------------------------------- safe metadata
class _NoGlobalsUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load global {module}.{name} from chunk metadata")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("refusing persistent id in chunk metadata")


def loads_chunks(data: bytes) -> list[tuple[str, str]]:
    obj = _NoGlobalsUnpickler(io.BytesIO(data)).load()
    if not isinstance(obj, list):
        raise IndexIntegrityError("chunk metadata is not a list")
    out = []
    for item in obj:
        if not (isinstance(item, (tuple, list)) and len(item) == 2
                and isinstance(item[0], str) and isinstance(item[1], str)):
            raise IndexIntegrityError("chunk metadata entries must be (source: str, text: str)")
        out.append((item[0], item[1]))
    return out


def dumps_chunks(chunks) -> bytes:
    return pickle.dumps([(str(s), str(t)) for s, t in chunks], protocol=4)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------- locking
@contextlib.contextmanager
def _lock(name: str, exclusive: bool):
    if fcntl is None:
        with _fallback_locks[name]:
            yield
        return
    path = os.path.join(LOCK_DIR, name)
    if exclusive:
        os.makedirs(LOCK_DIR, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    else:
        try:  # readers never create files; no lock file means no lock-aware writer exists yet
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            yield
            return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        os.close(fd)  # releases the flock


def build_lock():
    """Serialises whole index builds (across threads and processes)."""
    return _lock(BUILD_LOCK, exclusive=True)


def read_lock():
    return _lock(PUBLISH_LOCK, exclusive=False)


# ----------------------------------------------------------------------------- identities
def _identity(st: os.stat_result) -> list:
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]


def signature() -> tuple:
    """Cheap generation key: identities of the manifest and both artifacts plus the directory.

    Raises FileNotFoundError when no index is published (same as the original reader).
    """
    def stat_or_none(p):
        try:
            return tuple(_identity(os.stat(p)))
        except FileNotFoundError:
            return None
    index_st = os.stat(INDEX_PATH)
    chunks_st = os.stat(CHUNKS_PATH)
    return (os.path.realpath(INDEX_DIR), stat_or_none(MANIFEST_PATH),
            tuple(_identity(index_st)), tuple(_identity(chunks_st)))


def read_manifest() -> dict | None:
    try:
        with open(MANIFEST_PATH, "rb") as f:
            manifest = json.loads(f.read())
    except FileNotFoundError:
        return None
    except ValueError as e:
        raise IndexIntegrityError(f"unreadable manifest: {e}") from e
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise IndexIntegrityError(f"unsupported manifest schema {manifest.get('schema')!r}")
    return manifest


def _check_artifact(path: str, record: dict, data: bytes | None = None) -> None:
    """Fast path: inode/size/mtime identity recorded at publish. Fallback: content sha256
    (e.g. after a backup/restore changed inodes). Mismatch => IndexIntegrityError."""
    if list(_identity(os.stat(path))) == record["identity"]:
        return
    if data is None:
        with open(path, "rb") as f:
            data = f.read()
    if sha256_bytes(data) != record["sha256"]:
        COUNTERS["integrity_failures"] += 1
        raise IndexIntegrityError(f"{path} does not belong to the published generation; rebuild the index")


# ----------------------------------------------------------------------------- loading
class Generation:
    """One immutable, coherent view of the index. Never mutated after construction."""
    __slots__ = ("key", "index", "chunks", "manifest", "loaded_at", "load_ms", "nbytes")

    def __init__(self, key, index, chunks, manifest, load_ms):
        self.key, self.index, self.chunks, self.manifest = key, index, chunks, manifest
        self.loaded_at, self.load_ms = time.time(), load_ms
        self.nbytes = index.ntotal * index.d * 4 + sum(len(s) + len(t) for s, t in chunks)

    @property
    def generation_id(self):
        return self.manifest["generation"] if self.manifest else None


def load_generation() -> Generation:
    """Load the current generation coherently (shared publish lock + identity checks)."""
    t = time.perf_counter()
    with read_lock():
        key = signature()
        manifest = read_manifest()
        with open(CHUNKS_PATH, "rb") as f:
            chunk_bytes = f.read()
        if manifest is not None:
            _check_artifact(INDEX_PATH, manifest["artifacts"]["index"])
            _check_artifact(CHUNKS_PATH, manifest["artifacts"]["chunks"], chunk_bytes)
        index = faiss.read_index(INDEX_PATH)
        COUNTERS["index_disk_loads"] += 1
    chunks = loads_chunks(chunk_bytes)
    if index.ntotal != len(chunks):
        COUNTERS["integrity_failures"] += 1
        raise IndexIntegrityError(f"index has {index.ntotal} vectors but metadata has {len(chunks)} chunks")
    return Generation(key, index, chunks, manifest, (time.perf_counter() - t) * 1000)


def load_metadata() -> tuple[dict, list[tuple[str, str]]] | None:
    """Manifest + chunk texts of the current generation, for incremental builds.

    Returns None when there is nothing reusable (no index, legacy layout without a manifest,
    or an incoherent generation) -- callers then fall back to parsing every document.
    """
    try:
        with read_lock():
            manifest = read_manifest()
            if manifest is None:
                return None
            with open(CHUNKS_PATH, "rb") as f:
                chunk_bytes = f.read()
            _check_artifact(INDEX_PATH, manifest["artifacts"]["index"])
            _check_artifact(CHUNKS_PATH, manifest["artifacts"]["chunks"], chunk_bytes)
        COUNTERS["metadata_disk_loads"] += 1
        chunks = loads_chunks(chunk_bytes)
    except (FileNotFoundError, IndexIntegrityError, pickle.UnpicklingError, KeyError):
        return None
    if len(chunks) != manifest.get("n_chunks"):
        return None
    return manifest, chunks


# ----------------------------------------------------------------------------- publication
def _write_temp(final_path: str, data: bytes) -> str:
    tmp = f"{final_path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return tmp


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - e.g. platforms without directory fds
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(fd)


def _swap_in(pairs: list[tuple[str, str]]) -> None:
    """Replace every final path with its temp file, or roll all of them back.

    The committed artifacts are first hard-linked to backups (no data copy). If any rename
    fails, every final path is restored from its backup (or removed if it did not exist), so
    an in-process failure never leaves a torn generation. A hard crash mid-swap is detected
    by the manifest identity check on load (IndexIntegrityError) and repaired by a rebuild.
    """
    backups = {}
    try:
        for _, final in pairs:
            if os.path.exists(final):
                backup = f"{final}.bak-{os.getpid()}-{uuid.uuid4().hex}"
                os.link(final, backup)
                backups[final] = backup
        done = []
        try:
            for tmp, final in pairs:
                os.replace(tmp, final)
                done.append(final)
        except BaseException:
            for final in done:
                if final in backups:
                    os.replace(backups.pop(final), final)
                else:
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(final)
            raise
    finally:
        for backup in backups.values():
            with contextlib.suppress(FileNotFoundError):
                os.remove(backup)


def publish(chunks: list[tuple[str, str]], vectors: np.ndarray, documents: list[dict],
            embedding: dict, corpus_fingerprint: str, chunker: str | None = None) -> dict:
    """Atomically publish a new generation built from `chunks` and row-aligned `vectors`."""
    if len(chunks) != vectors.shape[0]:
        raise ValueError("chunks and vectors must be row-aligned")
    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype="float32"))
    index_bytes = faiss.serialize_index(index).tobytes()
    construct_ms = (time.perf_counter() - t0) * 1000
    chunk_bytes = dumps_chunks(chunks)
    os.makedirs(INDEX_DIR, exist_ok=True)
    temps = []
    try:
        tmp_index = _write_temp(INDEX_PATH, index_bytes)
        temps.append(tmp_index)
        tmp_chunks = _write_temp(CHUNKS_PATH, chunk_bytes)
        temps.append(tmp_chunks)
        previous = None
        try:
            previous = read_manifest()
        except IndexIntegrityError:
            pass
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "generation": (previous or {}).get("generation", 0) + 1,
            "generation_uuid": uuid.uuid4().hex,
            "created_unix": time.time(),
            "n_chunks": len(chunks),
            "dim": int(vectors.shape[1]),
            "metric": "L2",
            "index_type": "IndexFlatL2",
            "corpus_fingerprint": corpus_fingerprint,
            "chunker": chunker,
            "embedding": embedding,
            "documents": documents,
            "artifacts": {
                "index": {"sha256": sha256_bytes(index_bytes), "identity": _identity(os.stat(tmp_index))},
                "chunks": {"sha256": sha256_bytes(chunk_bytes), "identity": _identity(os.stat(tmp_chunks))},
            },
        }
        tmp_manifest = _write_temp(MANIFEST_PATH, json.dumps(manifest, indent=1).encode())
        temps.append(tmp_manifest)
        with _lock(PUBLISH_LOCK, exclusive=True):
            _swap_in([(tmp_index, INDEX_PATH), (tmp_chunks, CHUNKS_PATH), (tmp_manifest, MANIFEST_PATH)])
            temps.clear()
            _fsync_dir(INDEX_DIR)
        COUNTERS["publishes"] += 1
    finally:
        for p in temps:
            with contextlib.suppress(FileNotFoundError):
                os.remove(p)
    manifest["_timings_ms"] = {"index_construct": construct_ms, "publish_total": (time.perf_counter() - t0) * 1000}
    return manifest


def publish_empty() -> bool:
    """Withdraw the current generation (empty corpus). Returns True if anything was removed."""
    removed = False
    with _lock(PUBLISH_LOCK, exclusive=True):
        for p in (INDEX_PATH, CHUNKS_PATH, MANIFEST_PATH):
            with contextlib.suppress(FileNotFoundError):
                os.remove(p)
                removed = True
    if removed:
        COUNTERS["publishes"] += 1
    return removed


def verify() -> dict:
    """Offline integrity audit of the published generation (sha256 of every artifact)."""
    manifest = read_manifest()
    if manifest is None:
        raise IndexIntegrityError("no manifest")
    for name, path in (("index", INDEX_PATH), ("chunks", CHUNKS_PATH)):
        with open(path, "rb") as f:
            if sha256_bytes(f.read()) != manifest["artifacts"][name]["sha256"]:
                raise IndexIntegrityError(f"{name} content hash mismatch")
    return {"generation": manifest["generation"], "n_chunks": manifest["n_chunks"], "ok": True}
