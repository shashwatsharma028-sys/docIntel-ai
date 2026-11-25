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
