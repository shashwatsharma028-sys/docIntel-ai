"""Test-only helpers to synthesise PDFs with PyMuPDF. Not used by production code."""
from pathlib import Path

import fitz

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"
GOLDEN_DOCS = sorted(p.stem for p in GOLDEN_DIR.glob("*.txt"))


def make_pdf_bytes(pages: list[str], fontsize: float = 9) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()  # A4-ish default
        rect = fitz.Rect(36, 36, page.rect.width - 36, page.rect.height - 36)
        spare = page.insert_textbox(rect, text, fontsize=fontsize)
        if spare < 0:
            raise ValueError("fixture text does not fit on one page; split it into more pages")
    data = doc.tobytes()
    doc.close()
    return data


def write_pdf(path: Path, pages: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_pdf_bytes(pages))
    return path


def golden_text(stem: str) -> str:
    return (GOLDEN_DIR / f"{stem}.txt").read_text().strip()


def write_golden_pdfs(folder: Path) -> list[str]:
    names = []
    for stem in GOLDEN_DOCS:
        write_pdf(folder / f"{stem}.pdf", [golden_text(stem)])
        names.append(f"{stem}.pdf")
    return names


def numbered_words(n: int, prefix: str = "w") -> str:
    """'w0 w1 ... w{n-1}' -- word positions are recoverable from the text itself."""
    return " ".join(f"{prefix}{i}" for i in range(n))
