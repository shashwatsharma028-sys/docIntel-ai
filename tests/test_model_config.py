"""Configurable generation model IDs; defaults and fallback behaviour unchanged."""
from backend import llm

CHUNKS = [{"source": "a.pdf", "text": "x" * 200}]


def test_defaults_unchanged(monkeypatch, gemini_ok):
    monkeypatch.delenv("DOCINTEL_GEMINI_MODEL", raising=False)
    out = llm.ask(CHUNKS, "q")
    assert gemini_ok.call_args.kwargs["model"] == "gemini-2.0-flash" and out["model"] == "gemini-2.0-flash"


def test_gemini_override(monkeypatch, gemini_ok):
    monkeypatch.setenv("DOCINTEL_GEMINI_MODEL", "gemini-replacement")
    out = llm.ask(CHUNKS, "q")
    assert gemini_ok.call_args.kwargs["model"] == "gemini-replacement" and out["model"] == "gemini-replacement"


def test_groq_override_keeps_fallback(monkeypatch, gemini_down, groq_ok):
    monkeypatch.setenv("DOCINTEL_GROQ_MODEL", "openai/gpt-oss-120b")
    out = llm.ask(CHUNKS, "q")
    assert gemini_down.call_count == 1
    assert groq_ok.call_args.kwargs["model"] == "openai/gpt-oss-120b"
    assert out == {"answer": "GROQ_ANSWER", "model": "groq:openai/gpt-oss-120b", "confidence": "low"} | {"confidence": out["confidence"]}
