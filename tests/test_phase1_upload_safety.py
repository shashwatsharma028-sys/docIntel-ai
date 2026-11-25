"""Phase 1 targeted tests for the D-08 upload path fix."""
import pytest
from fastapi.testclient import TestClient

from backend import main
from tests.pdfutil import golden_text, make_pdf_bytes


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def _pdf():
    return make_pdf_bytes([golden_text("enterprise_leave_policy")])


@pytest.mark.parametrize("name,stored", [
    ("../../escape.pdf", "escape.pdf"),
    ("..\\..\\escape.pdf", "escape.pdf"),
    ("/etc/escape.pdf", "escape.pdf"),
    ("nested/dir/escape.pdf", "escape.pdf"),
])
def test_directory_components_are_stripped(client, embedder, isolated_workdir, name, stored):
    r = client.post("/upload", files={"file": (name, _pdf(), "application/pdf")})
    assert r.status_code == 200
    assert r.json() == {"message": f"{stored} uploaded and indexed successfully"}
    files = sorted(str(p.relative_to(isolated_workdir)) for p in isolated_workdir.rglob("*.pdf"))
    assert files == [f"data/pdfs/{stored}"]


@pytest.mark.parametrize("name", ["..", ".", "../", "a/.."])
def test_unusable_names_are_rejected_without_writing(client, embedder, isolated_workdir, name):
    r = client.post("/upload", files={"file": (name, _pdf(), "application/pdf")})
    assert r.status_code == 400
    assert not list(isolated_workdir.rglob("*.pdf"))
    assert not (isolated_workdir / "data" / "index").exists()


@pytest.mark.parametrize("name", ["", "..", "x/..", "bad\x00.pdf", None])
def test_safe_upload_path_rejects_unusable_names(isolated_workdir, name):
    # NUL bytes never survive the multipart layer, so this is checked on the helper directly.
    with pytest.raises(main.HTTPException) as exc:
        main.safe_upload_path(name)
    assert exc.value.status_code == 400


def test_symlinked_target_cannot_escape(client, embedder, isolated_workdir):
    pdfs = isolated_workdir / "data" / "pdfs"
    pdfs.mkdir(parents=True)
    (isolated_workdir / "outside.pdf").write_bytes(b"keep")
    (pdfs / "link.pdf").symlink_to(isolated_workdir / "outside.pdf")
    r = client.post("/upload", files={"file": ("link.pdf", _pdf(), "application/pdf")})
    assert r.status_code == 400
    assert (isolated_workdir / "outside.pdf").read_bytes() == b"keep"


def test_ordinary_filename_contract_unchanged(client, embedder):
    r = client.post("/upload", files={"file": ("Q3 report (final).pdf", _pdf(), "application/pdf")})
    assert r.json() == {"message": "Q3 report (final).pdf uploaded and indexed successfully"}


# ---------------------------------------------------------------- D-05 staged validation
import os  # noqa: E402

from backend import ingest  # noqa: E402


def test_malformed_upload_leaves_no_file_and_keeps_index(client, embedder, isolated_workdir):
    assert client.post("/upload", files={"file": ("good.pdf", _pdf(), "application/pdf")}).status_code == 200
    before = (isolated_workdir / "data" / "index" / "faiss.index").read_bytes()
    r = client.post("/upload", files={"file": ("bad.pdf", b"%PDF-1.4 garbage", "application/pdf")})
    assert r.status_code == 500  # unchanged error status
    assert os.listdir(isolated_workdir / "data" / "pdfs") == ["good.pdf"]
    assert (isolated_workdir / "data" / "index" / "faiss.index").read_bytes() == before


def test_malformed_same_name_replacement_keeps_original(client, embedder, isolated_workdir):
    good = _pdf()
    client.post("/upload", files={"file": ("doc.pdf", good, "application/pdf")})
    r = client.post("/upload", files={"file": ("doc.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 500
    assert (isolated_workdir / "data" / "pdfs" / "doc.pdf").read_bytes() == good


def test_failed_indexing_after_valid_upload_does_not_poison_later_uploads(
        client, strict_external_clients, embedder, isolated_workdir):
    calls = {"n": 0}
    real = embedder.embed_content

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated embedding outage")
        return real(**kw)

    strict_external_clients["ingest"].models.embed_content.side_effect = flaky
    assert client.post("/upload", files={"file": ("a.pdf", _pdf(), "application/pdf")}).status_code == 500
    r = client.post("/upload", files={"file": ("b.pdf", make_pdf_bytes([golden_text("legal_service_agreement")]),
                                                "application/pdf")})
    assert r.status_code == 200
    with open(ingest.CHUNKS_PATH, "rb") as f:
        import pickle
        assert sorted(s for s, _ in pickle.load(f)) == ["a.pdf", "b.pdf"]


def test_non_pdf_upload_behaviour_unchanged(client, embedder, isolated_workdir):
    # D-06 is NOT in Phase 1 scope: non-PDF files are still stored without validation.
    r = client.post("/upload", files={"file": ("notes.txt", b"hello " * 20, "text/plain")})
    assert r.status_code == 200
    assert os.listdir(isolated_workdir / "data" / "pdfs") == ["notes.txt"]
