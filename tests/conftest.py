"""Offline test harness for the DocIntel AI baseline suite.

Guarantees (enforced here, before any backend module is imported):
  * Real credentials are never read: GEMINI_API_KEY / GROQ_API_KEY are forced to dummy values.
  * The Gemini and Groq SDK client classes are replaced with mocks, so importing
    backend.ingest / backend.retriever / backend.llm constructs no real client.
  * Outbound network connections raise immediately (socket guard).
  * Every test runs in its own temporary working directory, because the backend
    uses relative paths (data/pdfs, data/index, data/audit_log.csv).
  * Each backend client attribute is replaced per test with a strict mock that
    FAILS on any un-configured call, so a test can only reach an "external API"
    through an explicitly injected fake.
"""
import os
import socket
from unittest import mock

import pytest

# --- credentials: never use real keys -------------------------------------------------
os.environ["GEMINI_API_KEY"] = "offline-test-key-not-real"
os.environ["GROQ_API_KEY"] = "offline-test-key-not-real"

# --- SDK clients: patched before backend import ---------------------------------------
_sdk_patchers = [mock.patch("google.genai.Client"), mock.patch("groq.Groq")]
for _p in _sdk_patchers:
    _p.start()

# --- network guard --------------------------------------------------------------------
_real_connect = socket.socket.connect


def _guarded_connect(self, address, *args, **kwargs):
    if self.family == getattr(socket, "AF_UNIX", object()):
        return _real_connect(self, address, *args, **kwargs)
    raise RuntimeError(f"Network access is forbidden in the offline test suite: {address!r}")


socket.socket.connect = _guarded_connect

from backend import ingest, llm, retriever  # noqa: E402  (must follow the patches above)
from tests.fakes import FakeEmbedder, strict_client  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_workdir(tmp_path, monkeypatch):
    """Run every test in an empty temp directory (backend paths are cwd-relative)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def strict_external_clients(monkeypatch):
    """Replace every module-level SDK client with a mock that fails if used un-configured."""
    clients = {
        "ingest": strict_client("ingest.client"),
        "retriever": strict_client("retriever.client"),
        "gemini": strict_client("llm.gemini_client"),
        "groq": strict_client("llm.groq_client"),
    }
    monkeypatch.setattr(ingest, "client", clients["ingest"])
    monkeypatch.setattr(retriever, "client", clients["retriever"])
    monkeypatch.setattr(llm, "gemini_client", clients["gemini"])
    monkeypatch.setattr(llm, "groq_client", clients["groq"])
    return clients


@pytest.fixture
def embedder(strict_external_clients):
    """Inject a deterministic offline embedder into BOTH ingest and retriever clients.

    The production get_embedding() functions still run; only the SDK call is faked.
    """
    fake = FakeEmbedder()
    strict_external_clients["ingest"].models.embed_content.side_effect = fake.embed_content
    strict_external_clients["retriever"].models.embed_content.side_effect = fake.embed_content
    return fake


@pytest.fixture
def gemini_ok(strict_external_clients):
    """Gemini returns a fixed answer; records calls."""
    m = strict_external_clients["gemini"].models.generate_content
    m.side_effect = None
    m.return_value = mock.Mock(text="GEMINI_ANSWER")
    return m


@pytest.fixture
def gemini_down(strict_external_clients):
    m = strict_external_clients["gemini"].models.generate_content
    m.side_effect = RuntimeError("simulated Gemini outage")
    return m


@pytest.fixture
def groq_ok(strict_external_clients):
    m = strict_external_clients["groq"].chat.completions.create
    m.side_effect = None
    m.return_value = mock.Mock(choices=[mock.Mock(message=mock.Mock(content="GROQ_ANSWER"))])
    return m


@pytest.fixture
def groq_down(strict_external_clients):
    m = strict_external_clients["groq"].chat.completions.create
    m.side_effect = RuntimeError("simulated Groq outage")
    return m
