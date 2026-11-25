import asyncio
import os
import weakref
import shutil
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from backend.ingest import build_index, parse_pdf, replace_and_build
from backend.retriever import retrieve
from backend.llm import ask

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    mode: str = "enterprise"

PDF_DIR = "data/pdfs"

# Blocking work (disk, FAISS, embedding and LLM SDK calls) runs in the thread pool so the event loop
# stays responsive; semaphores bound in-flight work, and therefore concurrent provider requests.
# Limits are PER PROCESS (per event loop): with N server workers, up to N x MAX_CONCURRENT_CHAT chat
# requests may call providers at once. Index builds are serialised HOST-WIDE by the flock build lock
# (at most one build's embedding calls at a time); there is no cross-host limit. PyMuPDF work runs on a
# single dedicated thread (backend/pdf_runtime.py). The google-genai and groq SDK clients are shared
# across threads; both send requests through httpx connection pools, whose thread safety we rely on
# (not independently verified for these SDK wrappers).
_MAX_LIMIT = 64


def _limit(var: str, default: int) -> int:
    raw = os.environ.get(var, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{var} must be an integer, got {raw!r}") from None
    if not 1 <= value <= _MAX_LIMIT:
        raise ValueError(f"{var} must be in 1..{_MAX_LIMIT}, got {value}")
    return value


MAX_CONCURRENT_CHAT = _limit("DOCINTEL_MAX_CONCURRENT_CHAT", 8)
MAX_CONCURRENT_UPLOADS = _limit("DOCINTEL_MAX_CONCURRENT_UPLOADS", 2)
_slots: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _slot(kind: str) -> asyncio.Semaphore:
    """Per-event-loop semaphores (an asyncio.Semaphore is bound to the loop that first waits on it)."""
    per_loop = _slots.setdefault(asyncio.get_running_loop(), {})
    if kind not in per_loop:
        per_loop[kind] = asyncio.Semaphore(MAX_CONCURRENT_CHAT if kind == "chat" else MAX_CONCURRENT_UPLOADS)
    return per_loop[kind]


def safe_upload_path(filename: str | None) -> tuple[str, str]:
    """D-08: keep only the final path component and confine the result to PDF_DIR."""
    name = os.path.basename((filename or "").replace("\\", "/"))
    if name in ("", ".", "..") or "\x00" in name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    root = os.path.realpath(PDF_DIR)
    path = os.path.realpath(os.path.join(root, name))
    if os.path.dirname(path) != root:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name, path


def _stage_upload(src, name) -> str:
    """D-05: write the upload under a name the indexer ignores and validate it; a PDF that cannot be
    parsed never enters the corpus (the error still propagates as HTTP 500)."""
    staged = os.path.join(PDF_DIR, f".upload-{uuid.uuid4().hex}.partial")
    try:
        with open(staged, "wb") as f:
            shutil.copyfileobj(src, f)
        if name.endswith(".pdf"):
            parse_pdf(staged)
    except BaseException:
        if os.path.exists(staged):
            os.remove(staged)
        raise
    return staged


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs(PDF_DIR, exist_ok=True)
    name, path = safe_upload_path(file.filename)
    async with _slot("upload"):
        staged = await run_in_threadpool(_stage_upload, file.file, name)
        try:
            # replacement + rebuild under one build-lock hold: a consistent corpus snapshot
            await run_in_threadpool(replace_and_build, staged, path)
        finally:
            if os.path.exists(staged):
                os.remove(staged)
    return {"message": f"{name} uploaded and indexed successfully"}


def _answer(req: ChatRequest) -> dict:
    try:
        chunks = retrieve(req.question)
        result = ask(chunks, req.question, req.mode)
        return {
            "answer": result["answer"],
            "model": result["model"],
            "confidence": result["confidence"],
            "sources": list(set(c["source"] for c in chunks))
        }
    except Exception as e:
        return {"answer": f"No documents indexed yet. Please upload a PDF first.", "model": "", "confidence": "low", "sources": []}


@app.post("/chat")
async def chat(req: ChatRequest):
    async with _slot("chat"):
        return await run_in_threadpool(_answer, req)

@app.get("/health")
async def health():
    return {"status": "ok"}