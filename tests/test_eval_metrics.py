"""Verifies the metric, BM25 and RRF definitions used in evaluation/run_retrieval_eval.py
(no models are loaded; ranking is stubbed)."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "run_retrieval_eval", Path(__file__).resolve().parent.parent / "evaluation" / "run_retrieval_eval.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


class StubSystems:
    def __init__(self, order):
        self.passages = [{"passage_id": f"P{i}"} for i in range(10)]
        self.order = order

    def rank(self, cfg, query):
        return self.order


def q(rel, neg=()):
    return {"qid": "Q", "question": "x", "relevant_passage_ids": rel, "hard_negative_passage_ids": list(neg),
            "query_types": ["factual"]}


def test_metric_definitions():
    res = ev.evaluate(StubSystems([3, 1, 7, 2, 0, 4, 5, 6, 8, 9]), {"kind": "bm25", "k1": 1.2, "b": 0.75}, [q(["P1", "P2", "P9"], ["P3"])])
    row = res["per_query"][0]
    assert row["precision@4"] == 2 / 4            # P1, P2 in top-4
    assert row["recall@4"] == pytest.approx(2 / 3)  # P9 outside top-4
    assert row["rr@10"] == 1 / 2 and row["first_relevant_rank"] == 2
    assert row["hard_negative_in_top4"] == ["P3"]


def test_mrr_is_zero_when_no_relevant_in_top10():
    order = list(range(10))
    systems = StubSystems(order)
    systems.passages.append({"passage_id": "P10"})
    assert ev.evaluate(systems, {"kind": "bm25", "k1": 1.2, "b": 0.75}, [q(["P10"])])["per_query"][0]["rr@10"] == 0.0


def test_rrf_fuses_by_reciprocal_rank():
    # doc 2 ranks 2nd and 1st -> highest fused score; ties broken by id
    assert ev.rrf([[0, 2, 1], [2, 1]], k=10)[:2] == [2, 1]
    assert ev.rrf([[0], [1]], k=60) == [0, 1]


def test_bm25_prefers_rarer_matching_terms_and_ignores_non_matches():
    bm = ev.BM25(["the hotel limit is 180 euros", "the meal allowance is 65 euros", "no overlap here"], 1.5, 0.75)
    s = bm.scores("hotel limit")
    assert s[0] > 0 and s[1] == 0 and s[2] == 0
    assert bm.scores("euros")[0] > 0 and bm.scores("euros")[2] == 0


def test_selection_rule_order():
    mk = lambda mrr, rec, lat: {"summary": {"rr@10": mrr, "recall@4": rec, "latency_p50_ms": lat}}
    assert ev.select([mk(0.9, 0.9, 5), mk(0.9, 1.0, 9), mk(0.8, 1.0, 1)]) is not None
    assert ev.select([mk(0.9, 0.9, 5), mk(0.9, 1.0, 9)])["summary"]["recall@4"] == 1.0
    assert ev.select([mk(0.9, 1.0, 5), mk(0.9, 1.0, 2)])["summary"]["latency_p50_ms"] == 2
