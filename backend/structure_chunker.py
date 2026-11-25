"""Optional structure-aware, token-aware PDF chunking (opt-in: DOCINTEL_CHUNKING=structure).

Same algorithm as configuration C evaluated in evaluation/chunking/ (T=192, no heading prefix):
PDF text blocks -> sections (a short block without a final period is a heading) -> paragraphs
packed within a section up to the token budget; oversize paragraphs are split into token
windows with 1/8 overlap; a heading is attached to its section's first chunk.

Token counts use a dependency-free WordPiece estimate (`estimate_tokens`). On the evaluation
corpus it never under-counted BGE tokens (max real chunk 179 tokens at budget 192); Gemini
tokenisation differs, so the budget is approximate for production embeddings.
"""
import math
import re

import fitz

TOKEN_BUDGET = 192
_PIECE = re.compile(r"[A-Za-z]+|\d|[^\w\s]")


def estimate_tokens(text: str) -> int:
    return sum(max(1, math.ceil(len(p) / 7)) if p.isalpha() else 1 for p in _PIECE.findall(text))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _windows(words, budget, overlap, count):
    wt = [max(1, count(w)) for w in words]
    out, i = [], 0
    while i < len(words):
        j, used = i, 0
        while j < len(words) and (used + wt[j] <= budget or j == i):
            used += wt[j]
            j += 1
        out.append(" ".join(words[i:j]))
        if j >= len(words):
            break
        back, k = 0, j
        while k > i + 1 and back + wt[k - 1] <= overlap:
            k -= 1
            back += wt[k]
        i = k
    return out


def _is_heading(block: str) -> bool:
    return len(block.split()) <= 12 and not block.endswith(".")


def _pdf_blocks(pdf):
    doc = fitz.open(stream=pdf, filetype="pdf") if isinstance(pdf, (bytes, bytearray)) else fitz.open(pdf)
    return [_norm(b[4]) for page in doc for b in page.get_text("blocks") if b[6] == 0 and _norm(b[4])]


def chunk_pdf_structured(pdf_path: "str | bytes", budget: int = TOKEN_BUDGET, count=estimate_tokens) -> list[str]:
    from backend.pdf_runtime import run_pdf  # PyMuPDF only on its dedicated thread
    blocks = run_pdf(_pdf_blocks, pdf_path)
    sections, cur = [], None
    for b in blocks:
        if _is_heading(b):
            cur = {"head": b, "paras": []}
            sections.append(cur)
        else:
            if cur is None:
                cur = {"head": "", "paras": []}
                sections.append(cur)
            cur["paras"].append(b)
    return chunk_sections([(s["head"], s["paras"]) for s in sections], budget, count)


def chunk_sections(sections, budget: int = TOKEN_BUDGET, count=estimate_tokens) -> list[str]:
    """sections: [(heading or "", [paragraph, ...]), ...] -> chunks (shared by the PDF path)."""
    chunks = []
    for head, paras in sections:
        s = {"head": head, "paras": paras}
        # reserve room for the heading so the first chunk of a section stays within budget
        room = budget - (count(s["head"] + ". ") if s["head"] else 0)
        pieces = []
        for p in s["paras"]:
            pieces += [p] if count(p) <= room else _windows(p.split(), room, room // 8, count)
        if s["head"]:
            pieces = [s["head"] + ". " + pieces[0]] + pieces[1:] if pieces else [s["head"]]
        current = ""
        for p in pieces:
            candidate = (current + " " + p).strip()
            if current and count(candidate) > budget:
                chunks.append(current)
                current = p
            else:
                current = candidate
        if current:
            chunks.append(current)
    return chunks
