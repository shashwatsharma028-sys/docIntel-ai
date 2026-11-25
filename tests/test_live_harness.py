"""Offline checks of the live-pilot harness safety layer (mock transport; no network)."""
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

spec = importlib.util.spec_from_file_location("le", Path(__file__).resolve().parent.parent / "evaluation/qasper/live_eval.py")
le = importlib.util.module_from_spec(spec)
spec.loader.exec_module(le)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(le, "LEDGER", tmp_path / "ledger.jsonl")
    le.STATE.update(headers={}, quota_checked=False)
    return le


def mock_client(le, statuses):
    from openai import OpenAI
    seq = iter(statuses)

    def handler(request):
        status = next(seq)
        body = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Yes"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}} if status == 200 else \
               {"error": {"message": "server error"}}
        return httpx.Response(status, json=body, headers={"x-ratelimit-remaining-requests": "500"})
    http = httpx.Client(transport=httpx.MockTransport(handler),
                        event_hooks={"request": [le._before_request], "response": [le._sync_response]})
    return OpenAI(base_url=le.BASE_URL, api_key="placeholder", http_client=http, max_retries=0)


def call(cl):
    return cl.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}], max_completion_tokens=100)


def test_every_attempt_including_failures_and_retries_is_counted(harness, monkeypatch):
    monkeypatch.setattr(harness.time, "sleep", lambda s: None)
    cl = mock_client(harness, [500, 500, 200])
    harness.with_retries(lambda: call(cl))
    n, used = harness.ledger_stats()
    assert n == 3 and used == 15
    rows = [json.loads(l) for l in harness.LEDGER.read_text().splitlines()]
    assert [r["status"] for r in rows if r["event"] == "response"] == [500, 500, 200]


def test_budget_is_enforced_before_dispatch(harness, monkeypatch):
    monkeypatch.setattr(harness, "MAX_TOTAL_TOKENS", 50)  # reservation = 1//3 + 100 > 50
    cl = mock_client(harness, [200])
    with pytest.raises(harness.BudgetStop):
        harness.with_retries(lambda: call(cl))  # must surface unwrapped, not be retried
    assert harness.ledger_stats() == (0, 0)  # nothing was sent


def test_request_cap(harness, monkeypatch):
    monkeypatch.setattr(harness, "MAX_REQUESTS", 1)
    cl = mock_client(harness, [200, 200])
    call(cl)
    with pytest.raises(harness.BudgetStop):
        harness.with_retries(lambda: call(cl))
    assert harness.ledger_stats()[0] == 1


def test_quota_rule_stops_when_insufficient(harness):
    harness.STATE["headers"] = {"x-ratelimit-remaining-requests": "100"}
    with pytest.raises(harness.BudgetStop):
        harness.quota_check(1)  # needs 1.25 x (79 + 76)
    harness.STATE["headers"] = {"x-ratelimit-remaining-requests": "990"}
    harness.quota_check(1)


def test_trace_sanitiser_drops_secrets_and_clients():
    d = le._sanitize({"question": "q", "api_key": "k", "client": object(), "Authorization": "b", "contexts": ["c"]})
    assert set(d) == {"question", "contexts"}
