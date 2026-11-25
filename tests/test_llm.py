"""Baseline tests for backend/llm.py: domain prompts, fallback, confidence, citations, audit log."""
import csv
import datetime

import pytest

from backend import llm

MODES = ["enterprise", "legal", "medical", "study"]
CHUNKS = [
    {"source": "policy.pdf", "text": "Annual leave accrues at 1.5 days per month."},
    {"source": "contract.pdf", "text": "Either party may terminate with 90 days notice."},
    {"source": "policy.pdf", "text": "Unused leave up to five days carries forward."},
]


def read_audit():
    with open("data/audit_log.csv", newline="") as f:
        return list(csv.reader(f))


# ---------------------------------------------------------------- domain modes
class TestDomainPrompts:
    def test_exactly_four_modes(self):
        assert set(llm.SYSTEM_PROMPTS) == set(MODES)

    @pytest.mark.parametrize("mode,marker", [
        ("enterprise", "enterprise knowledge assistant"),
        ("legal", "legal document analyst"),
        ("medical", "medical report explainer"),
        ("study", "study assistant"),
    ])
    def test_prompt_identity(self, mode, marker):
        assert marker in llm.SYSTEM_PROMPTS[mode]

    @pytest.mark.parametrize("mode", MODES)
    def test_every_prompt_requires_citation(self, mode):
        assert "Always cite the source document" in llm.SYSTEM_PROMPTS[mode]

    def test_domain_specific_guardrails(self):
        assert "Never diagnose" in llm.SYSTEM_PROMPTS["medical"]
        assert "⚠️" in llm.SYSTEM_PROMPTS["legal"]
        assert "ONLY the provided company documents" in llm.SYSTEM_PROMPTS["enterprise"]
        assert "flashcard" in llm.SYSTEM_PROMPTS["study"]

    @pytest.mark.parametrize("mode", MODES)
    def test_gemini_receives_mode_prompt_then_context(self, mode, gemini_ok):
        llm.ask(CHUNKS, "Q?", mode)
        kwargs = gemini_ok.call_args.kwargs
        assert kwargs["model"] == "gemini-2.0-flash"
        assert kwargs["contents"] == f"{llm.SYSTEM_PROMPTS[mode]}\n\n{llm.build_prompt(CHUNKS, 'Q?')}"

    @pytest.mark.parametrize("mode", MODES)
    def test_groq_receives_mode_prompt_as_system_message(self, mode, gemini_down, groq_ok):
        llm.ask(CHUNKS, "Q?", mode)
        kwargs = groq_ok.call_args.kwargs
        assert kwargs["model"] == "llama-3.3-70b-versatile"
        assert kwargs["max_tokens"] == 1000
        assert kwargs["messages"] == [
            {"role": "system", "content": llm.SYSTEM_PROMPTS[mode]},
            {"role": "user", "content": llm.build_prompt(CHUNKS, "Q?")},
        ]

    def test_unknown_mode_falls_back_to_enterprise_prompt(self, gemini_ok):
        llm.ask(CHUNKS, "Q?", "astrology")
        assert gemini_ok.call_args.kwargs["contents"].startswith(llm.SYSTEM_PROMPTS["enterprise"])

    def test_default_mode_is_enterprise(self, gemini_ok):
        llm.ask(CHUNKS, "Q?")
        assert gemini_ok.call_args.kwargs["contents"].startswith(llm.SYSTEM_PROMPTS["enterprise"])
        assert read_audit()[0][1] == "enterprise"


# ---------------------------------------------------------------- source citations in prompt
class TestBuildPrompt:
    def test_exact_format(self):
        chunks = [{"source": "a.pdf", "text": "AAA"}, {"source": "b.pdf", "text": "BBB"}]
        assert llm.build_prompt(chunks, "Why?") == (
            "Context:\n[Source: a.pdf]\nAAA\n\n[Source: b.pdf]\nBBB\n\nQuestion: Why?")

    def test_every_chunk_is_labelled_with_its_source_in_order(self):
        prompt = llm.build_prompt(CHUNKS, "q")
        positions = [prompt.index(f"[Source: {c['source']}]\n{c['text']}") for c in CHUNKS]
        assert positions == sorted(positions)

    def test_empty_chunks(self):
        assert llm.build_prompt([], "q") == "Context:\n\n\nQuestion: q"


# ---------------------------------------------------------------- fallback
class TestFallback:
    def test_gemini_success_does_not_call_groq(self, gemini_ok, strict_external_clients):
        out = llm.ask(CHUNKS, "Q?", "legal")
        assert out["answer"] == "GEMINI_ANSWER"
        assert out["model"] == "gemini-2.0-flash"
        strict_external_clients["groq"].chat.completions.create.assert_not_called()

    def test_gemini_failure_falls_back_to_groq(self, gemini_down, groq_ok, capsys):
        out = llm.ask(CHUNKS, "Q?", "medical")
        assert out["answer"] == "GROQ_ANSWER"
        assert out["model"] == "groq-llama-3.3"
        assert gemini_down.call_count == 1 and groq_ok.call_count == 1
        assert "falling back to Groq" in capsys.readouterr().out

    def test_both_providers_failing_raises_and_writes_no_audit_row(self, gemini_down, groq_down):
        with pytest.raises(RuntimeError, match="Groq outage"):
            llm.ask(CHUNKS, "Q?")
        import os
        assert not os.path.exists("data/audit_log.csv")

    def test_ask_return_shape(self, gemini_ok):
        assert set(llm.ask(CHUNKS, "Q?")) == {"answer", "model", "confidence"}


# ---------------------------------------------------------------- confidence
class TestConfidenceContract:
    @pytest.mark.parametrize("chunks", [[], CHUNKS, [{"source": "x", "text": "y" * 1000}]])
    def test_value_domain(self, chunks):
        assert llm.get_confidence(chunks) in {"high", "medium", "low"}

    def test_no_evidence_is_low(self):
        assert llm.get_confidence([]) == "low"

    def test_ask_reports_confidence_of_retrieved_chunks(self, gemini_ok):
        assert llm.ask(CHUNKS, "Q?")["confidence"] == llm.get_confidence(CHUNKS)


@pytest.mark.characterization
class TestConfidenceHeuristicCharacterization:
    """Pins the CURRENT length-based heuristic (avg chars > 300 high, > 150 medium) so that an
    optimisation cannot change it silently. This is NOT the desired design -- see D-02."""

    @staticmethod
    def _chunks(*lengths):
        return [{"source": "s", "text": "x" * n} for n in lengths]

    @pytest.mark.parametrize("lengths,expected", [
        ((301,), "high"), ((300,), "medium"), ((151,), "medium"), ((150,), "low"), ((0,), "low"),
        ((600, 0), "medium"), ((602, 0), "high"), ((100, 100, 400), "medium"),
    ])
    def test_thresholds(self, lengths, expected):
        assert llm.get_confidence(self._chunks(*lengths)) == expected


class TestConfidenceKnownDefects:
    @pytest.mark.known_defect
    @pytest.mark.xfail(strict=True, reason="D-02: confidence depends only on chunk length, "
                       "not on query relevance or retrieval distance")
    def test_irrelevant_long_chunks_are_not_high_confidence(self):
        irrelevant = [{"source": "cookbook.pdf", "text": "Whisk the eggs and sugar. " * 20}] * 4
        assert llm.get_confidence(irrelevant) != "high"


# ---------------------------------------------------------------- audit logging
class TestAuditLog:
    def test_row_written_per_query(self, gemini_ok):
        before = datetime.datetime.now()
        llm.ask(CHUNKS, "What is the leave policy?", "enterprise")
        rows = read_audit()
        assert len(rows) == 1
        ts, mode, question, answer, sources = rows[0]
        assert before <= datetime.datetime.fromisoformat(ts) <= datetime.datetime.now()
        assert (mode, question, answer) == ("enterprise", "What is the leave policy?", "GEMINI_ANSWER...")
        assert sorted(sources.split(", ")) == ["contract.pdf", "policy.pdf"]

    def test_answer_truncated_to_100_chars_plus_ellipsis(self, gemini_ok):
        gemini_ok.return_value.text = "A" * 250
        llm.ask(CHUNKS, "q", "study")
        assert read_audit()[0][3] == "A" * 100 + "..."

    def test_rows_append_and_csv_quoting_round_trips(self, gemini_ok):
        llm.ask(CHUNKS, 'first, "quoted"\nquestion', "legal")
        llm.ask(CHUNKS, "second", "medical")
        rows = read_audit()
        assert [r[1:3] for r in rows] == [["legal", 'first, "quoted"\nquestion'], ["medical", "second"]]
        assert all(len(r) == 5 for r in rows)

    def test_no_header_row(self, gemini_ok):
        llm.ask(CHUNKS, "q")
        assert read_audit()[0][1] == "enterprise"

    def test_fallback_queries_are_logged(self, gemini_down, groq_ok):
        llm.ask(CHUNKS, "q", "study")
        assert read_audit()[0][3] == "GROQ_ANSWER..."

    def test_logged_sources_are_deduplicated(self, gemini_ok):
        llm.ask(CHUNKS, "q")
        assert sorted(read_audit()[0][4].split(", ")) == ["contract.pdf", "policy.pdf"]
