"""Integrity of the frozen Phase 2 evaluation corpus and labels (no retrieval is evaluated here)."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parent.parent / "evaluation"
LOCK = json.loads((EVAL / "LABELS.lock.json").read_text())
PASSAGES = {json.loads(l)["passage_id"]: json.loads(l) for l in (EVAL / "corpus/passages.jsonl").read_text().splitlines()}
QUESTIONS = [json.loads(l) for split in ("dev", "heldout")
             for l in (EVAL / f"questions/{split}.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("rel", sorted(LOCK["sha256"]))
def test_frozen_files_unchanged(rel):
    assert hashlib.sha256((EVAL / rel).read_bytes()).hexdigest() == LOCK["sha256"][rel], \
        f"{rel} changed after the labels were frozen"


def test_counts_match_lock():
    c = LOCK["counts"]
    assert len(PASSAGES) == c["passages"]
    assert Counter(q["split"] for q in QUESTIONS) == {"dev": c["dev_questions"], "heldout": c["heldout_questions"]}
    assert 40 <= len(QUESTIONS) <= 60


def test_ids_unique_and_splits_disjoint():
    qids = [q["qid"] for q in QUESTIONS]
    assert len(qids) == len(set(qids))
    assert len({q["question"].lower() for q in QUESTIONS}) == len(QUESTIONS)


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q["qid"])
def test_labels_are_well_formed(q):
    rel, neg = set(q["relevant_passage_ids"]), set(q["hard_negative_passage_ids"])
    assert rel and rel <= set(PASSAGES) and neg <= set(PASSAGES) and not rel & neg
    if "multi_passage" in q["query_types"]:
        assert len(rel) >= 2
    if "distractor" in q["query_types"]:
        assert neg, "distractor questions must name their hard negatives"
    if "numerical" in q["query_types"]:
        evidence = " ".join(PASSAGES[p]["text"] for p in rel)
        number = r"\d+(?:\.\d+)?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"
        nums = re.findall(number, q["reference_answer"].lower())
        assert nums and any(n in evidence.lower() for n in nums), "numerical answer not grounded in its relevant passages"


def test_passages_appear_in_their_documents():
    for p in PASSAGES.values():
        assert p["text"] in (EVAL / "corpus" / f"{p['doc_id']}.txt").read_text()


def test_query_type_coverage():
    types = Counter(t for q in QUESTIONS for t in q["query_types"])
    for t in ("numerical", "paraphrase", "distractor", "multi_passage", "factual"):
        assert types[t] >= 5, (t, types)
