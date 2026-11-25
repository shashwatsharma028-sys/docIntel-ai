import os
import shutil
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.ingest import build_index, parse_pdf
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


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs(PDF_DIR, exist_ok=True)
    name, path = safe_upload_path(file.filename)
    # D-05: stage under a name the indexer ignores; a PDF that cannot be parsed never enters
    # the corpus (and never replaces an existing good file). The error still propagates (HTTP 500).
    staged = os.path.join(PDF_DIR, f".upload-{uuid.uuid4().hex}.partial")
    try:
        with open(staged, "wb") as f:
            shutil.copyfileobj(file.file, f)
        if name.endswith(".pdf"):
            parse_pdf(staged)
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    build_index()
    return {"message": f"{name} uploaded and indexed successfully"}

@app.post("/chat")
async def chat(req: ChatRequest):
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

@app.get("/health")
async def health():
    return {"status": "ok"}