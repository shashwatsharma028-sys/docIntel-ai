"""Optional structure-aware chunking (DOCINTEL_CHUNKING=structure). Default behaviour is unchanged."""
import pickle

import fitz
import pytest

from backend import index_store, ingest, retriever, structure_chunker
from tests.pdfutil import golden_text


def sectioned_pdf(path, sections):
    pdf = fitz.open()
    page, y = pdf.new_page(), 40
    for head, paras in sections:
        for text, size in [(head, 12)] + [(p, 9) for p in paras]:
            rect = fitz.Rect(40, y, page.rect.width - 40, page.rect.height - 40)
            spare = page.insert_textbox(rect, text, fontsize=size)
            if spare < 0:
                page, y = pdf.new_page(), 40
                rect = fitz.Rect(40, y, page.rect.width - 40, page.rect.height - 40)
                spare = page.insert_textbox(rect, text, fontsize=size)
            y += rect.height - spare + 8
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(path)
    return path


SECTIONS = [("Leave Policy Overview", [golden_text("enterprise_leave_policy")]),
            ("Service Agreement Terms", [golden_text("legal_service_agreement")]),
            ("Laboratory Findings Summary", [golden_text("medical_blood_report"), "Follow up in six months is advised."])]


@pytest.fixture
def structured(monkeypatch):
    monkeypatch.setenv("DOCINTEL_CHUNKING", "structure")


def test_default_strategy_is_unchanged(isolated_workdir, embedder, monkeypatch):
    monkeypatch.delenv("DOCINTEL_CHUNKING", raising=False)
    pdf = sectioned_pdf(isolated_workdir / "data/pdfs/a.pdf", SECTIONS)
    ingest.build_index()
    with open(ingest.CHUNKS_PATH, "rb") as f:
        assert [t for _, t in pickle.load(f)] == ingest.chunk_text(ingest.parse_pdf(str(pdf)))


def test_unknown_strategy_is_rejected(monkeypatch):
    monkeypatch.setenv("DOCINTEL_CHUNKING", "magic")
    with pytest.raises(ValueError):
        ingest.chunker_signature()


def test_chunks_respect_budget_sections_and_cover_all_text(tmp_path):
    pdf = sectioned_pdf(tmp_path / "s.pdf", SECTIONS)
    chunks = structure_chunker.chunk_pdf_structured(str(pdf))
    assert all(structure_chunker.estimate_tokens(c) <= structure_chunker.TOKEN_BUDGET for c in chunks)
    heads = [h for h, _ in SECTIONS]
    for c in chunks:  # a chunk never spans two sections
        assert sum(h in c for h in heads) <= 1
    assert [c for c in chunks if any(c.startswith(h + ".") for h in heads)].__len__() == 3
    joined = " ".join(chunks).split()
    for _, paras in SECTIONS:
        for p in paras:
            assert set(p.split()) <= set(joined)


def test_oversize_paragraph_is_split_with_overlap(tmp_path):
    long_para = " ".join(f"term{i} detail" for i in range(300)) + "."
    pdf = sectioned_pdf(tmp_path / "l.pdf", [("Long Section Heading", [long_para])])
    chunks = structure_chunker.chunk_pdf_structured(str(pdf))
    assert len(chunks) > 2 and all(structure_chunker.estimate_tokens(c) <= 192 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.split()[-5:]) & set(b.split()[:30]), "consecutive windows must overlap"


def test_structure_strategy_end_to_end_and_strategy_switch_reuses_embeddings(
        isolated_workdir, embedder, structured, monkeypatch):
    sectioned_pdf(isolated_workdir / "data/pdfs/a.pdf", SECTIONS)
    stats = ingest.build_index()
    with open(ingest.CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    assert len(chunks) >= 3 and {s for s, _ in chunks} == {"a.pdf"}
    assert "structure" in index_store.read_manifest()["chunker"]
    top = retriever.retrieve("haemoglobin ferritin anaemia", top_k=1)[0]
    assert top["source"] == "a.pdf" and "anaemia" in top["text"]
    again = ingest.build_index()
    assert again["unchanged"] and stats["embeddings_computed"] == len(chunks)
    monkeypatch.setenv("DOCINTEL_CHUNKING", "words")  # switching strategy re-chunks the corpus
    assert ingest.build_index()["documents_parsed"] == 1
    monkeypatch.setenv("DOCINTEL_CHUNKING", "structure")  # switching back reuses cached embeddings
    back = ingest.build_index()
    assert back["documents_parsed"] == 1 and back["embeddings_computed"] == 0
